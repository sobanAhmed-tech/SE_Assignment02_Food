import streamlit as st
import pandas as pd
import ollama
import logging

# Configure logging
logging.basicConfig(filename='llama3_streamlit.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset (Ensure the correct path)
try:
    df = pd.read_csv("recipe.csv")
except FileNotFoundError:
    st.error("⚠️ The file 'recipe.csv' was not found. Please upload the correct file.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="Recipe Query Assistant", layout="wide")
st.title("🍽️ Recipe Query Assistant")
st.write("Ask about recipes, ingredients, cooking times, and more!")

# User input
user_query = st.text_input("Enter your query:", "")


# Function to generate query code from Ollama
def generate_query_code(query):
    try:
        response = ollama.chat(
            model='llama3',
            messages=[
                {
                    'role': 'system',
                    'content': '''You are a Python expert that generates **ONLY valid Python code** to query a Pandas DataFrame (`df`).  
    Return **ONLY** executable Python code without any explanation or markdown formatting.  
    
    The DataFrame `df` has these columns:
    - RecipeId, Barcode, Name, AuthorId, AuthorName, CookTime, PrepTime, TotalTime
    - DatePublished, Description, Images, RecipeCategory, Keywords
    - RecipeIngredientQuantities, RecipeIngredientParts, AggregatedRating
    - ReviewCount, Calories, FatContent, SaturatedFatContent, CholesterolContent
    - SodiumContent, CarbohydrateContent, FiberContent, SugarContent, ProteinContent
    - RecipeServings, RecipeYield, RecipeInstructions
    
    **Rules:**
    - **Do NOT drop rows.** Use `.fillna('')` instead.
    - **Always use `.str.strip().fillna('')` when selecting text columns.**
    - **Output must be stored in `result`**, e.g., `result = df[...]`
    - **Do NOT include explanations or comments.**  
    '''
                },
                {
                    'role': 'user',
                    'content': f"Generate Python code to query the recipe dataset based on: {query}. "
                               "Ensure all selected columns use `.fillna('')` to avoid errors. "
                               "Use case-insensitive search when filtering text, like `.str.contains('...', case=False)`. "
                               "Store the output in `result`.",
                }
            ]
        )
        return response.get('message', {}).get('content', '').strip()
    except Exception as e:
        logging.error(f"Error in generating query: {e}")
        return ""


# Function to execute generated query safely
def execute_query_code(code):
    if "result =" not in code.lower():
        logging.error("Generated invalid code. Expected 'result = ...'")
        return None

    exec_locals = {"df": df}
    try:
        exec(code, {}, exec_locals)
        return exec_locals.get("result", None)
    except Exception as e:
        logging.error(f"Error executing code: {e}")
        return None


# Function to get LLM response if no data is found
def get_llm_response(query):
    try:
        response = ollama.chat(
            model='llama3',
            messages=[
                {
                    'role': 'system',
                    'content': "You are an AI assistant that provides detailed and accurate recipe information."
                },
                {
                    'role': 'user',
                    'content': f"Please provide a detailed recipe for: {query}"
                }
            ]
        )
        return response.get('message', {}).get('content', '')
    except Exception as e:
        logging.error(f"Error in getting LLM response: {e}")
        return "⚠️ Sorry, we couldn't find the recipe, and the AI couldn't generate a response at this time."


# Main logic
if st.button("Generate Query & Extract Info") and user_query:
    with st.spinner("Generating query..."):
        generated_code = generate_query_code(user_query)

        if not generated_code:
            st.warning("⚠️ Failed to generate query. Please try again.")
        else:
            logging.info(f"Generated code:\n{generated_code}")
            st.subheader("Generated Query Code:")
            st.code(generated_code, language="python")

            result = execute_query_code(generated_code)

            # **Retry if execution fails**
            if result is None:
                logging.info("Retrying query generation due to failure...")
                generated_code = generate_query_code(user_query)
                result = execute_query_code(generated_code)

            if result is not None and not result.empty:
                st.success("Query executed successfully! 🎉")
                st.dataframe(result)

                # Extract insights using LLM
                with st.spinner("Extracting key insights..."):
                    try:
                        text_response = ollama.chat(
                            model='llama3',
                            messages=[
                                {
                                    'role': 'system',
                                    'content': "You are an assistant that extracts key information from structured data. "
                                               "Summarize key insights from the given dataset."
                                },
                                {
                                    'role': 'user',
                                    'content': f"Summarize key insights from this data:\n{result.to_string(index=False)}"
                                }
                            ]
                        )

                        extracted_text = text_response.get('message', {}).get('content', '')
                        if extracted_text:
                            st.subheader("📌 Key Insights:")
                            st.write(extracted_text)
                        else:
                            st.warning("⚠️ No insights generated.")

                    except Exception as e:
                        logging.error(f"Error extracting insights: {e}")
                        st.warning("⚠️ Unable to extract insights at the moment.")

            else:
                # **Fallback: Ask LLM for Recipe**
                with st.spinner("No recipe found in the dataset. Asking AI for help..."):
                    llm_response = get_llm_response(user_query)
                    st.subheader("🔍 AI-Generated Recipe:")
                    st.write(llm_response)
                    
"""

CSTU Chatbot
GPT Application Project
Team 2 - Akshat, Asma, Atilla, Maheswar, Kavya, Vishal

- Environment Variable Loading: The code loads environment variables from a specified file to retrieve the OpenAI and Pinecone API keys.
- JSON Data: The code includes hardcoded JSON data for courses, FAQs, jobs, and addresses.
- Embedding Function: The get_embeddings function generates embeddings for input texts using OpenAI's embedding model.
- Upsert Data: The upsert_data function inserts data into the Pinecone index, creating embeddings for each piece of data.
- Chat Completion: The chat_complete_messages function generates a response from the OpenAI model based on input messages.
- Query Pinecone: The get_relevant_info function retrieves relevant information from the Pinecone index based on a query.
- Respond Function: The respond function updates the chat history and generates responses.
- Gradio Interface: The code uses Gradio to create a simple chat interface for interacting with the chatbot.
- Main Function: The main function sets up the Pinecone index, inserts data, and runs the chatbot interface.

"""

import os
import json
from dotenv import load_dotenv
import openai
from pinecone import Pinecone
import gradio as gr

# Load environment variables
load_dotenv()

# Specify the path to the environment variable file
dotenv_path = '/Users/ak/.config/chatgpt/chatgpt.env'
load_dotenv(dotenv_path)

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

index = pinecone_client.Index("cstu-bot")

# JSON data directly defined within the code
json_data = {
    "Course_Details": [
        {
            "code": "MB/CSE 652",
            "title": "Prompt Engineering",
            "details": "Explore advanced techniques for understanding and generating human language."
        },
        {
            "code": "MB/CSE 632",
            "title": "Introduction to Cloud Computing",
            "details": "Dive into the world of on-demand computing resources with this introduction to Cloud Computing! Learn how to leverage the cloud for storage, processing, and applications."
        },
        {
            "code": "MB/CSE 648",
            "title": "GPT application",
            "details": "Unleash the power of AI: Learn to leverage GPT for automatic topic extraction, improve content organization, and unlock deeper insights from your text."
        },
        {
            "code": "MB/CSE 600",
            "title": "Python",
            "details": "Python empowers you to automate tasks, analyze data, and build applications. Learn Python from scratch or enhance your skills in this versatile programming language."
        },
        {
            "code": "MB/CSE 590",
            "title": "Network Security and AI",
            "details": "This course explores how Artificial Intelligence empowers network security strategies, defending against evolving cyber threats."
        },
        {
            "code": "MB/CSE 638",
            "title": "Deep learning with TensorFlow",
            "details": "Master deep learning concepts and build neural networks for real-world tasks using TensorFlow, a popular open-source framework."
        }
    ],
    "International_Students_faqs": [
        {
            "question": "What are the different types of fees for international students?",
            "answer": "There are several fees associated with studying at CSTU University for international students. These include:\n* Technology Fee: $50 (one-time fee)\n* Students Association Fee: $40 (per semester)\n* Registration Fee: $250 (one-time fee)\n* Graduation Fee: $250 (one-time fee)\n* Per Credit Fee: $700\n**Note:** Each Course has 1.5 Credits and lasts one Term (2 months)."
        },
        {
            "question": "What payment methods are available for international students?",
            "answer": "International students can pay their fees through the following methods:\n* **Credit Card:** Accepted credit cards include Visa, Mastercard, and American Express. A secure online payment portal is available within the university system.\n* **Bank Transfer:** International students can initiate a bank transfer to the university designated account. Details and instructions will be provided upon request."
        },
        {
            "question": "When are the deadlines for fee payments?",
            "answer": "Fee payment deadlines are typically set before the start of each semester. Specific deadlines will be communicated to students via email and are also available on the university website. Timely payments are crucial to avoid late fees and potential registration holds."
        },
        {
            "question": "What happens if I have questions about my student account or need assistance with payments?",
            "answer": "The university International Student Office is here to help! You can contact them via email at iso@cstu.edu. They can assist with any questions or concerns regarding your student account and fee payments."
        },
        {
            "question": "What is Curricular Practical Training (CPT)?",
            "answer": "Curricular Practical Training (CPT) is an authorization for F-1 international students to gain practical experience directly related to their major field of study. It must be an integral part of your academic program and requires approval from your Designated School Official (DSO). There are limitations on the duration of CPT, so be sure to consult the International Student Office for details and eligibility requirements."
        },
        {
            "question": "What is Optional Practical Training (OPT)?",
            "answer": "Optional Practical Training (OPT) allows F-1 students temporary employment authorization to gain practical experience after completing their academic studies. There are two types of OPT: pre-completion (before graduation) and post-completion (after graduation). Students must apply for OPT through USCIS with approval from their DSO. STEM (Science, Technology, Engineering, and Mathematics) program graduates may be eligible for an extended OPT period."
        },
        {
            "question": "What is SEVIS (Student and Exchange Visitor Information System)?",
            "answer": "The Student and Exchange Visitor Information System (SEVIS) is a U.S. Department of Homeland Security (DHS) database that tracks F and M nonimmigrant students enrolled in U.S. schools. All educational institutions approved to enroll international students must report student information to SEVIS. Maintaining a valid SEVIS record is crucial for your immigration status as an international student."
        },
        {
            "question": "Where can I find more information about CPT, OPT, and SEVIS?",
            "answer": "We recommend consulting the following resources for more detailed information on CPT, OPT, and SEVIS:\n* CSTU University International Student Office website\n* U.S. Department of Homeland Security Study in the States website (https://studyinthestates.dhs.gov/assets/SEVP_InternationalStudentLifecycle_2022-BW.pdf)."
        },
        {
            "question": "What are the deadlines for the application?",
            "answer": "The deadlines for the application varies with each term, Here are the Deadlines for the Undergraduation and Post-Graduation Summer term: Apirl 1st 2024,\n* Fall term: June 1st 2024,\n* Spring term: September 1st 2024."
        }
    ],
    "On_Campus_Jobs": [
        {
            "job_title": "Teaching Assistant",
            "description": "Supports professors with courses, leads discussions & grades assignments. Eligibility: Enrolled students with strong subject knowledge & excellent communication skills."
        },
        {
            "job_title": "Research Assistant",
            "description": "Aids researchers in experiments, data collection & analysis. Eligibility: Students with research experience, strong analytical skills & attention to detail."
        }
    ],
    "Addresses": [
        {
            "name": "University Location",
            "address": "California Science and Technology University (CSTU), 100 Innovation Way, Santa Clara, CA 95054",
            "google_maps_link": "https://www.google.com/maps/place/California+Science+and+Technology+University+(CSTU)+100+Innovation+Way,+Santa+Clara,+CA+95054"
        },
        {
            "name": "Medical Services or Hospitals",
            "address": "Santa Clara Valley Medical Center, 751 S. Wells Fargo Dr, Santa Clara, CA 95054",
            "google_maps_link": "https://www.google.com/maps/place/Santa+Clara+Valley+Medical+Center,+751+S+Wells+Fargo+Dr,+Santa+Clara,+CA+95054"
        },
        {
            "name": "Police Services",
            "address": "Santa Clara Police Department - Main Station, 150 Santa Clara Ave, Santa Clara, CA 95050",
            "google_maps_link": "https://www.google.com/maps/place/Santa+Clara+Police+Department+-+Main+Station,+150+Santa+Clara+Ave,+Santa+Clara,+CA+95050"
        },
        {
            "name": "Coffee Shop",
            "address": "200 Innovation Way, Santa Clara, CA 95054",
            "google_maps_link": "https://www.google.com/maps/place/Coffee+Shop+Example+(placeholder)+200+Innovation+Way,+Santa+Clara,+CA+95054"
        },
        {
            "name": "Restaurant",
            "address": "35 Main St, Santa Clara, CA 95050",
            "google_maps_link": "https://www.google.com/maps/place/Restaurant+Example+(placeholder)+35+Main+St,+Santa+Clara,+CA+95050"
        }
    ]
}

# chatContext data defined within the code
chatContext = [
    {'role':'system', 'content': f"""
Objective: You are a smart, friendly virtual CSTU Chatbot Assistent, assisting students with their quires and assiting with knowing information on courses, International Students faqs, Locations, and Jobs.
Procedure:
1. Greet the student and inquire about their reason for contacting you.
2. If the student asks about fees, proceed with step 3.
3. Identify the student's program and semester to retrieve their specific fee structure.
4. Explain the fee structure clearly, including details like tuition, technology fees, and any additional program-specific fees.
5. Offer options for viewing a detailed breakdown of fees (e.g., link to university portal).
6. If the student wants to pay fees, guide them through the available payment methods (e.g., credit card, bank transfer).
7. For online payments, offer a secure and user-friendly interface within the LLM system (if possible) or direct the student to a secure university payment portal.
8. For other payment methods, provide clear instructions and relevant links (e.g., bank account details for transfer).
9. After successful payment, provide confirmation and offer a receipt for download or email delivery.

Task Steps:
1. When a student inquires about fees, ask for their program and semester to personalize the response.
2. Based on the provided information, retrieve the relevant fee structure from the university database.
3. Present the fee structure in a clear and concise format, highlighting key elements like total fees, due dates, and payment methods.
4. If the student chooses to pay online, guide them through the secure payment interface within the LLM (if available) or redirect them to the university's secure payment portal.
5. For offline payments, provide detailed instructions and relevant information (e.g., bank account details).
6. Once payment is processed, confirm the success with the student and offer a downloadable or emailed receipt.

The final step:
1. For online payments within the LLM system, ensure a secure and successful transaction with proper confirmation and receipt generation.
2. Regardless of the payment method, provide a clear overview of the student's remaining balance, if any.
3. Offer additional resources like FAQs or links to the university's financial aid office for further assistance.

[Fee structure:]
The fee structure includes:
- Technology fee: $50
- Students Association fee: $40
- Registration fee: $250
- Graduation fee(one time payment): $250
- Per Credit fee: $700
- note: Each Course has 1.5 Credit, Duration: one Term (2 months)

Procedure to search data in the Pinecone index (cstu-bot):
1. To retrieve course details, international student FAQs, locations, or jobs information, specify the metadata type (e.g., 'Course_Details').
2. Use the Pinecone index 'cstu-bot' to query the relevant information based on the student's request.
3. Retrieve and present the requested data to the student.
"""}
]

# Example user inputs for testing and debugging
example_inputs = [
    "Tell me about the Python class",  # Should match Course_Details
    "How much are the fees for international students?",  # Should match International_Students_faqs
    "Where is the university located?",  # Should match Addresses
    "Are there any jobs available on campus?",  # Should match On_Campus_Jobs
]

def get_embeddings(texts):
    """
    Generate embeddings for a list of texts using OpenAI's embedding model.
    
    Args:
        texts (list): A list of strings to generate embeddings for.
    
    Returns:
        list: A list of embeddings corresponding to the input texts.
    """
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Use the appropriate embedding model
        input=texts
    )
    return [embedding['embedding'] for embedding in response['data']]

def upsert_data(index):
    """
    Upsert data into the Pinecone index.
    
    Args:
        index (pinecone.Index): The Pinecone index to upsert data into.
    """
    for metadata_type, data in json_data.items():
        vectors_to_upsert = []

        if metadata_type == "Course_Details":
            for doc in data:
                text = doc['title'] + " " + doc['details']
                doc_id = doc['code']
                vectors_to_upsert.append((doc_id, text))

        elif metadata_type == "International_Students_faqs":
            for doc in data:
                text = doc['answer']
                doc_id = doc['question']
                vectors_to_upsert.append((doc_id, text))

        elif metadata_type == "On_Campus_Jobs":
            for doc in data:
                text = doc['description']
                doc_id = doc['job_title']
                vectors_to_upsert.append((doc_id, text))

        elif metadata_type == "Addresses":
            for item in data:
                text = item['name'] + " " + item['address']
                doc_id = item['name']
                vectors_to_upsert.append((doc_id, text))

        else:
            continue  # Skip unrecognized metadata types

        # Upsert vectors to Pinecone
        for doc_id, text in vectors_to_upsert:
            embedding = get_embeddings([text])[0]
            index.upsert(vectors=[(doc_id, embedding, {"type": metadata_type, "text": text})])
            print(f"Document '{doc_id}' from {metadata_type} has been upserted into Pinecone successfully.")

def chat_complete_messages(messages, temperature=0.7):
    """
    Generate a response from the OpenAI model based on the input messages.
    
    Args:
        messages (list): A list of message dictionaries in the format required by the OpenAI API.
        temperature (float): The temperature setting for the model (default is 0.7).
    
    Returns:
        str: The response content generated by the model.
    """
    completion = openai.ChatCompletion.create(
        model="gpt-4",  # Or use "gpt-4" if available for your use case
        messages=messages,
        temperature=temperature
    )
    return completion.choices[0].message['content']

def get_relevant_info(index, query, metadata_type, top_k=5):
    """
    Query the Pinecone index for relevant information based on the input query.
    
    Args:
        index (pinecone.Index): The Pinecone index to query.
        query (str): The query string to search for.
        metadata_type (str): The type of metadata to filter results by.
        top_k (int): The number of top results to return (default is 5).
    
    Returns:
        list: A list of filtered query results matching the specified metadata type.
    """
    query_embedding = get_embeddings([query])[0]

    query_result = index.query(
        vector=query_embedding,
        top_k=top_k,  # Number of top results to return
        include_metadata=True
    )

    # Filter results to ensure they match the metadata_type
    filtered_results = [
        match for match in query_result['matches']
        if match['metadata']['type'] == metadata_type
    ]

    return filtered_results

def chat_complete_messages_gr(text):
    """
    Generate a response from the OpenAI model for a given text input.
    
    Args:
        text (str): The input text from the user.
    
    Returns:
        str: The response content generated by the model.
    """
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": text}],
        temperature=0,  # This is the degree of randomness of the model's output
    )
    return completion.choices[0].message['content']

def format_chat_prompt(message, chat_history):
    """
    Format the chat history and the current user message into a single prompt.
    
    Args:
        message (str): The current user message.
        chat_history (list): A list of tuples representing the chat history.
    
    Returns:
        str: The formatted prompt including the chat history and the current message.
    """
    prompt = ""
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def respond(message, chat_history):
    """
    Generate a response to a user message and update the chat history.
    
    Args:
        message (str): The current user message.
        chat_history (list): The chat history up to this point.
    
    Returns:
        tuple: An empty string and the updated chat history.
    """
    chatContext.append({'role': 'user', 'content': message})
    response = chat_complete_messages(chatContext)
    chat_history.append((message, response))
    chatContext.append({'role': 'assistant', 'content': response})
    return "", chat_history

def main():
    """
    Main entry point of the app.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("cstu-bot")
    
    # Call the function to upsert data
    upsert_data(index)

    # Test the examples
    for example in example_inputs:
        print(f"User Input: {example}")
        # Add user input to chat context
        chatContext.append({'role': 'user', 'content': example})

        # Determine the type of information the user is requesting
        metadata_type = None
        if any(keyword in example.lower() for keyword in ["course", "class", "module", "study"]):
            metadata_type = "Course_Details"
        elif any(keyword in example.lower() for keyword in ["faq", "fees", "payment", "international student"]):
            metadata_type = "International_Students_faqs"
        elif any(keyword in example.lower() for keyword in ["location", "place", "map", "address"]):
            metadata_type = "Addresses"
        elif any(keyword in example.lower() for keyword in ["job", "position", "work", "assistant"]):
            metadata_type = "On_Campus_Jobs"

        if metadata_type:
            # Query Pinecone for nearest neighbors based on user input
            query_results = get_relevant_info(index, example, metadata_type)

            if query_results:
                response_message_content = f"I found some relevant information about {metadata_type}:\n"
                for match in query_results:
                    response_message_content += f"- {match['metadata']['text']} (from {match['metadata']['type']})\n"
            else:
                response_message_content = f"I'm sorry, I couldn't find any relevant information about {metadata_type}."
        else:
            # Use OpenAI to generate responses if the type is not recognized
            response_message_content = chat_complete_messages(chatContext)

        print("ChatBot: ", response_message_content)
        chatContext.append({'role': 'assistant', 'content': response_message_content})

    # Main chatbot loop (for interactive use)
    print("Welcome! I'm here to assist you with information about courses, International Students FAQs, locations, and jobs.")

    def on_clear():
        """
        Clear the chat history.
        
        Returns:
            tuple: An empty value update for the message input and an empty list for the chat history.
        """
        global chat_history
        chat_history = []
        return gr.update(value=""), []

    with gr.Blocks() as myDemo:
        chatbot = gr.Chatbot(height=600)
        msg = gr.Textbox(label="Please input for Chatting")
        btn = gr.Button("Send")
        clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

        btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])  # Press enter to submit
        clear.click(on_clear, outputs=[msg, chatbot])

    myDemo.launch(share=True, inbrowser=True)

if __name__ == "__main__":
    main()

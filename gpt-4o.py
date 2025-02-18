import requests
import base64
import json
from mimetypes import guess_type
import os

def convert_image_to_base64(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found
    # Read and encode the image file
    with open(image_path, 'rb') as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def send_multimodal_request(api_key, image_url, messages):
    # Prepare the headers and data
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o",  # Use a model that supports vision
        "stream": False,
        "max_tokens": 1000,
        "messages": messages
    }
    
    # Send the request
    response = requests.post(
        "https://models-proxy.stepfun-inc.com/v1/chat/completions",  # Use the correct API endpoint
        headers=headers,
        data=json.dumps(data)
    )
    
    # Return the response
    return response.json()

def save_answers_to_file(answers, file_path):
    # Save only the answers to a JSON file
    with open(file_path, 'w') as file:
        json.dump(answers, file, indent=4)

# Example usage
api_key = "ak-39d8efgh45i6jkl23mno78pqrs12tuv4k5"
# image_folder = "/data/caiweiwei/kohya_ss/data/mesh_image/1_data"
image_folder = 'mesh_images'
output_folder = "gpt_outputs/textdeformer_images_json"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# List all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png'))]

# Questions to ask
questions = [
    "Knowing that the above is a 2D image rendered from a 3x3 view of a 3D mesh, describe the object in the image with a description statement that includes: This image depicts a 2D mesh of stylized xxx captured from a xxx view. Then describe the details of the object. For example:“The image depicts a 2D mesh of a stylized whale, captured from a side perspective. The mesh features a massive, elongated body with a distinct head. The whale's head is characterized by a large, curved beak and a pair of eyes, contributing to its majestic appearance. The body is streamlined and tapers towards the tail, which is broad and flat. The mesh exhibits a faceted, low-polygon style, which gives the whale a geometric and simplified look. The absence of texture and color in the mesh suggests that it is a basic 3D model, potentially serving as a foundation for further design and development.”.",
    "Give an text description of the similar object, describe the similar object with a description statement that includes: This image depicts a 2D mesh of stylized xxx captured from a xxx view. Then describe the details of the object. For example:“The image depicts a 2D mesh of a stylized whale, captured from a side perspective. The mesh features a massive, elongated body with a distinct head. The whale's head is characterized by a large, curved beak and a pair of eyes, contributing to its majestic appearance. The body is streamlined and tapers towards the tail, which is broad and flat. The mesh exhibits a faceted, low-polygon style, which gives the whale a geometric and simplified look. The absence of texture and color in the mesh suggests that it is a basic 3D model, potentially serving as a foundation for further design and development.”."
]

# Process each image
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image_url = convert_image_to_base64(image_path)
    
    # Initialize the conversation with the image and the first question
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": questions[0]
                }
            ]
        }
    ]
    
    # List to store answers for this image
    answers = []
    
    # Process each question
    for i, question in enumerate(questions):
        response = send_multimodal_request(api_key, image_url, messages)
        print(f"Response to question {i+1} for {image_file}:", response)
        
        # Extract the assistant's answer and add it to the answers list
        answer = response['choices'][0]['message']['content']
        answers.append({
            "question": question,
            "answer": answer
        })
        
        # Add the assistant's response to the messages for the next question
        messages.append({
            "role": "assistant",
            "content": answer
        })
        
        # Add the next question to the messages
        if i < len(questions) - 1:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": questions[i+1]
                    }
                ]
            })
    
    # Save the answers for this image to a single JSON file
    output_file_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_answers.json")
    save_answers_to_file(answers, output_file_path)
    print(f"Answers for {image_file} saved to {output_file_path}")
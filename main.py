from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from pydantic import BaseModel
from mysql.connector import connect, Error
from bcrypt import hashpw, gensalt

# MYSQL
# MySQL connection configuration
db_config = {
    "user": "root",
    "password": "a:}a8BU5{aa02GY/",
    "host": "hello-fish-3ef3c:asia-southeast2:hello-fish-database",
    "database": "app-database",
}

# Connect to MySQL
try:
    conn = connect(**db_config)
    print("Connected to MySQL")
except Error as err:
    print(f"Error connecting to MySQL: {err}")


# MODEL
# Load the trained model
model = load_model('path/to/model.h5')

# Define the target image size expected by the model
target_size = (224, 224)

# preprocess image
def preprocess_image(image):
    img = image.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img


app = FastAPI()


# START
@app.get("/")
async def root():
    return {"message": "Hallo FISHHH!"}


# APP without login
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file
    image = Image.open(io.BytesIO(await file.read()))

    # Preprocess the image
    img = preprocess_image(image)

    # Perform the prediction
    pred = model.predict(img)
    class_indices = {0: "WHITESPOT", 1: "BACTERIALFIN", 2: "REDSPOT", 3: "HEALTHY"}
    predicted_class = class_indices[np.argmax(pred)]
    confidence = np.max(pred) * 100

    return {"predicted_class": predicted_class, "confidence": confidence}


# USERS
class UserSignUp(BaseModel):
    username: str
    password: str
    email: str

# signup
@app.post("/signup")
def signup(user_data: UserSignUp):
    try:
        # Hash the password
        hashed_password = hashpw(user_data.password.encode(), gensalt())
        
        # Insert user data into the database
        query = "INSERT INTO users (username, password, email) VALUES (%s, %s, %s)"
        values = (user_data.username, hashed_password, user_data.email)
        
        with conn.cursor() as cursor:
            cursor.execute(query, values)
            conn.commit()
        
        return {"message": "User sign-up successful"}
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=str(err))





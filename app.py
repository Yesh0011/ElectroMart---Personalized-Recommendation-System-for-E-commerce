#IRWA Reccomendation System
#IT22888716 , IT , IT, IT

#------------------------------------------------------------------------------------------------------

#Import Libraies

from flask import Flask, request, render_template, session, redirect, url_for

import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#------------------------------------------------------------------------------------------------------


app = Flask(__name__)

# Database configuration

app.secret_key = "SLIIT"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:SLIIT@localhost/electromart"  
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

#------------------------------------------------------------------------------------------------------


# Define your model class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    userName = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(10), nullable=False)

#------------------------------------------------------------------------------------------------------


# Function to get content-based recommendations

def content_based_recommendations(train_data, product_name, top_n=5):

    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english') # Vectorize the 'Tags' or 'Description' column

    if 'Tags' in train_data.columns:
        train_data['Tags'] = train_data['Tags'].fillna('')  # Fill missing values
        tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['Tags'])
    else:
        raise ValueError("The 'Tags' column is missing from the dataset")

    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix) # Compute the cosine similarity 
    indices = pd.Series(train_data.index, index=train_data['Name']).drop_duplicates() # Find the index of the product that matches the product_name

    
    if product_name not in indices:
        return pd.DataFrame()  # Return an empty DataFrame if product not found

    idx = indices[product_name]
    session['searched_product'] = product_name


    sim_scores = list(enumerate(cosine_sim[idx])) # Get pairwise similarity scores for all products with the input product
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Sort the products based on similarity scores
    sim_scores = sim_scores[1:top_n+1] # Get the scores of the top_n most similar products

    
    product_indices = [i[0] for i in sim_scores] # Get the product indices
    return train_data.iloc[product_indices][['Name', 'Rating', 'ReviewCount', 'ImageURL']]

#---------------------------------------------------------------------------------------------------------


# Function for collaborative filtering

def collaborative_filtering(user_id, train_data, top_n=20):
    
    user_item_matrix = train_data.pivot_table(index='UserID', columns='ProdID', values='Rating').fillna(0) #matrix creation
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    # Get the similar users for the given user_id
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:top_n + 1]
    recommended_items = user_item_matrix.loc[similar_users].mean(axis=0).sort_values(ascending=False).head(top_n) # Get the products rated by similar users
    
    # Return product details for the recommended items
    recommended_products = train_data[train_data['ProdID'].isin(recommended_items.index)]
    return recommended_products[['Name', 'Rating', 'ReviewCount', 'ImageURL']]

#----------------------------------------------------------------------------------------------------------------------------------

#Hybrid Filtering

def hybrid_filtering(user_id, product_name, train_data, top_n=20, content_weight=0.5, collab_weight=0.5):

    # Get content-based recommendations
    try:
        content_recommendations = content_based_recommendations(train_data, product_name, top_n)
        content_recommendations['score'] = content_weight
    except Exception as e:
        print(f"Content-based filtering error: {e}")
        content_recommendations = pd.DataFrame()

    # Get collaborative filtering recommendations
    try:
        collaborative_recommendations = collaborative_filtering(user_id, train_data, top_n)
        collaborative_recommendations['score'] = collab_weight
    except Exception as e:
        print(f"Collaborative filtering error: {e}")
        collaborative_recommendations = pd.DataFrame()

    # Combine the recommendations (if both are available)
    if not content_recommendations.empty and not collaborative_recommendations.empty:
        combined_recommendations = pd.concat([content_recommendations, collaborative_recommendations]).drop_duplicates(subset='Name')
    elif not content_recommendations.empty:
        combined_recommendations = content_recommendations
    elif not collaborative_recommendations.empty:
        combined_recommendations = collaborative_recommendations
    else:
        combined_recommendations = pd.DataFrame()

    # Sort based on the score (higher is better)
    if not combined_recommendations.empty:
        combined_recommendations['final_score'] = combined_recommendations['score'].fillna(0)  # Ensure there's a score
        combined_recommendations = combined_recommendations.sort_values('final_score', ascending=False).head(top_n)

    return combined_recommendations[['Name', 'Rating', 'ReviewCount', 'ImageURL']]

#---------------------------------------------------------------------------------------------------------------------

#routes

@app.route("/")
def welcome():
    
    train_data = pd.read_csv('models/cleaned_data_edited.csv') # Load your cleaned CSV file

    train_data['Rating'] = pd.to_numeric(train_data['Rating'], errors='coerce') # Convert the 'Rating' column to numeric, forcing errors to NaN

    # Calculate average ratings and sort
    average_ratings = train_data.groupby(['Name', 'ReviewCount', 'ImageURL', 'Category', 'Description', 'Tags'])['Rating'].mean().reset_index()
    top_rated_items = average_ratings.sort_values(by='Rating', ascending=False).head(15)
    rating_base_recommendation = top_rated_items.to_dict(orient='records') # Convert to a dictionary for easier access in the template

   
    return render_template('welcome.html', top_rated_items=rating_base_recommendation)

#-------------------------------------------------------------------------------------------------------------------------------

@app.route("/index")
def index():
    
    train_data = pd.read_csv('models/cleaned_data_edited.csv')
    train_data['Rating'] = pd.to_numeric(train_data['Rating'], errors='coerce')

    
    average_ratings = train_data.groupby(['Name', 'ReviewCount', 'ImageURL', 'Category', 'Description', 'Tags'])['Rating'].mean().reset_index()
    top_rated_items = average_ratings.sort_values(by='Rating', ascending=False).head(15)
    rating_base_recommendation = top_rated_items.to_dict(orient='records')

    user_id = session.get('user_id')  # Get the logged-in user's ID
    product_name = session.get('searched_product') #if available

    if user_id and product_name:
        # Apply hybrid filtering
        hybrid_recommendations = hybrid_filtering(user_id, product_name, train_data)
        hybrid_recommendations = hybrid_recommendations.to_dict(orient='records')
    else:
        hybrid_recommendations = None

    
    return render_template('index.html', hybrid_recommendations=hybrid_recommendations, top_rated_items=rating_base_recommendation)

#------------------------------------------------------------------------------------------------------------------------------------

@app.route("/main", methods=['GET', 'POST'])
def main():
    recommendations = None
    product_name = None
    if request.method == 'POST':
        product_name = request.form['product_name'] #get user inputs
        num_results = int(request.form['num_results'])

        
        train_data = pd.read_csv('models/cleaned_data_edited.csv')
        recommendations = content_based_recommendations(train_data, product_name, top_n=num_results)
        recommendations = recommendations.to_dict(orient='records') # Convert the DataFrame to a dictionary for rendering

    return render_template('main.html', recommendations=recommendations, product_name=product_name)

#-------------------------------------------------------------------------------------------------------------------------------


@app.route("/signupform", methods=['POST','GET'])
def signupform():
    if request.method == 'POST':
        userName = request.form['userName']  
        email = request.form['email']
        password = request.form['password']

        # Create a new signup object
        new_signup = Signup(userName=userName, email=email, password=password)

        # Add the new record to the database and commit it
        db.session.add(new_signup)
        db.session.commit()

        
        session['user_id'] = new_signup.id  # Save user ID in session

        return redirect(url_for('index', success='true'))

    return index()


#-----------------------------------------------------------------------------------------------------------------------------------------


@app.route("/signinform", methods=['GET', 'POST'])
def signinform():
    if request.method == 'POST':
        userName = request.form['userName']
        password = request.form['password']

        # Query the database for the user
        user = Signup.query.filter_by(userName=userName).first() 

        if user and user.password == password:  
            session['user_id'] = user.id  
            return redirect(url_for('index', success='true'))

        else:
            error_message = "Invalid username or password." # Invalid credentials, return error message
            return render_template('signin.html', error=error_message)
        
#-----------------------------------------------------------------------------------------------------------------------------------------

@app.route("/collaborative")
def collaborative():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('signin'))

    train_data = pd.read_csv('models/cleaned_data_edited.csv') 
    train_data['UserID'] = train_data['ID']  # Replace with actual user IDs
    
    recommendations = collaborative_filtering(user_id, train_data)
    recommendations = recommendations.to_dict(orient='records')

    return render_template('collaborative.html', recommendations=recommendations)

#-------------------------------------------------------------------------------------------------------------------------------

@app.route("/hybrid")
def hybrid():
    user_id = session.get('user_id')
    product_name = session.get('searched_product')
    
    if not user_id or not product_name:
        return redirect(url_for('signin'))

    # Load your cleaned data
    train_data = pd.read_csv('models/cleaned_data_edited.csv')

    # Get hybrid recommendations
    recommendations = hybrid_filtering(user_id, product_name, train_data)
    recommendations = recommendations.to_dict(orient='records')

    return render_template('hybrid.html', recommendations=recommendations)

#---------------------------------------------------------------------------------------------------------------------

@app.route("/signin")
def signin():
    return render_template('signin.html')

#---------------------------------------------------------------------------------------------------------------------

@app.route("/signup")
def signup():
    return render_template('signup.html')

#---------------------------------------------------------------------------------------------------------------------

@app.route("/logout")
def logout():
    session.pop('user_id', None)  # Remove user ID from session
    session.pop('searched_product', None)
    return welcome()  # Redirect to welcome page or index

#---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
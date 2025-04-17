# Import necessary modules from Django and standard Python libraries
from django.shortcuts import render, redirect # For rendering templates and handling redirects
from django.http import HttpResponse # For returning simple HTTP responses (not used directly here but often useful)
from django.views.decorators.cache import cache_control # Decorator to control browser caching
import os # For interacting with the operating system, specifically for path manipulation
import joblib # For loading pre-trained machine learning models saved with joblib

# --- Model Loading ---
# Construct the absolute path to the model files relative to the current file's directory
# This ensures the models are found regardless of where the script is run from
model_dir = os.path.dirname(__file__)
model1_path = os.path.join(model_dir, "mySVCModel1.pkl") # Path for the first model (likely Support Vector Classifier)
model2_path = os.path.join(model_dir, "myModel.pkl")    # Path for the second model (type unspecified in filename)

# Load the pre-trained machine learning models from the .pkl files
# These models are expected to have a 'predict' method
model1 = joblib.load(model1_path)
model2 = joblib.load(model2_path)

# --- View Functions ---

# Decorator to prevent caching of this view's response in the browser or intermediate caches.
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def index(request):
    """
    Handles requests for the main page ('/').
    Renders the main application page ('index.html') where users can input text.
    Authentication is no longer required.
    """
    # Always render the main index page, regardless of GET or POST (no login form anymore)
    return render(request, 'index.html')

# Decorator to prevent caching of this view's response.
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def checkSpam(request):
    """
    Handles POST requests to the '/check' URL for classifying input text as spam or not.
    Requires the user to be authenticated.
    Uses one of two pre-loaded models based on user selection.
    """
    # Check if the request method is POST (form submission from index.html)
    if request.method == "POST":
        # Retrieve the raw text data input by the user from POST data
        rawData = request.POST.get("rawdata")

        # --- Model Selection and Prediction ---
        prediction = None # Initialize prediction variable
        # Use the first loaded model (model1) to predict
        # Note: model expects a list/iterable of texts, hence [rawData]
        prediction = model1.predict([rawData])[0] # Get the first prediction result
        # Render the output page ('output.html') displaying the prediction
        return render(request, 'output.html', {'prediction': prediction, 'rawdata': rawData})
    # If the request method is GET (user tries to access '/check' directly via URL)
    else:
        # Redirect GET requests to the main page, as checking requires a POST submission
        return redirect('/')


# Decorator to prevent caching of this view's response.
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def logout(request):
    """
    Handles requests to the '/logout' URL.
    Clears the user's session data, effectively logging them out.
    Redirects the user to the main/login page.
    """
    # Check if the user is currently logged in (session key exists)
    if request.session.get('authdetails'): # Check if the key exists and has a value
        print("Logging out user...") # Debugging print statement
        # Clear all data stored in the current session
        request.session.clear()
        # request.session.flush() # Use flush() to delete session data from backend storage immediately
        print("Session cleared.") # Debugging print statement
        # Redirect the user to the main page (which will show the login form as they are now logged out)
        return redirect('/')
    else:
        # If the user wasn't logged in anyway, just redirect them to the main page
        return redirect('/')

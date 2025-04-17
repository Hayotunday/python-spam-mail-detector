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
    # This ensures that authentication status and content are always fresh.
    @cache_control(no_cache=True, must_revalidate=True, no_store=True)
    def index(request):
        """
        Handles requests for the main page ('/') and the authentication path ('/auth').
        If the user is authenticated (session exists), it shows the main index page.
        If not authenticated, it shows the login page ('auth.html').
        Handles POST requests for login attempts.
        """
        # Check if the request method is POST (user submitted the login form)
        if request.method == "POST":
            # Retrieve username and password from the POST data
            un = request.POST.get('username')
            up = request.POST.get('password')
    
            # --- Simple Hardcoded Authentication ---
            # Check if the provided username and password match the hardcoded credentials
            if un == "timi" and up == "timi":
                # If credentials are correct, store an authentication marker in the session
                request.session['authdetails'] = "timi"
                # Double-check if the session was set correctly (optional but present)
                if request.session.get('authdetails') == "timi":
                    # Render the main application page ('index.html')
                    return render(request, 'index.html')
                else:
                    # If session setting failed unexpectedly, redirect back to the auth page
                    return redirect('/auth') # Or render('auth.html') might be better
            else:
                # If credentials do not match, re-render the login page ('auth.html')
                # Optionally, add an error message to the context here
                return render(request, 'auth.html')
        # If the request method is GET (user is navigating to the page)
        else:
            # Check if the user is already authenticated by looking for 'authdetails' in the session
            if request.session.get('authdetails') == "timi": # More robust check than has_key
                print("Session Auth: User already logged in.") # Debugging print statement
                # Render the main application page ('index.html')
                return render(request, 'index.html')
            else:
                # If not authenticated, render the login page ('auth.html')
                return render(request, 'auth.html')
    
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
            # --- Authentication Check ---
            # Verify if the user is authenticated by checking the session
            if request.session.get('authdetails') == "timi":
                # Retrieve the selected algorithm ('Algo-1' or 'Algo-2') from POST data
                algo = request.POST.get("algo")
                # Retrieve the raw text data input by the user from POST data
                rawData = request.POST.get("rawdata")
    
                # --- Model Selection and Prediction ---
                prediction = None # Initialize prediction variable
                if algo == "Algo-1":
                    # Use the first loaded model (model1) to predict
                    # Note: model expects a list/iterable of texts, hence [rawData]
                    prediction = model1.predict([rawData])[0] # Get the first prediction result
                    # Render the output page ('output.html') displaying the prediction
                    return render(request, 'output.html', {"answer": prediction})
                elif algo == "Algo-2":
                    # Use the second loaded model (model2) to predict
                    prediction = model2.predict([rawData])[0] # Get the first prediction result
                    # Render the output page ('output.html') displaying the prediction
                    return render(request, 'output.html', {"answer": prediction})
                else:
                    # Handle cases where the 'algo' value is unexpected (optional)
                    # Maybe redirect back with an error message
                    return redirect('/') # Or render index with an error
            else:
                # If the user is not authenticated, redirect them to the main/login page
                return redirect('/')
        # If the request method is GET (user tries to access '/check' directly via URL)
        else:
            # Redirect GET requests to the main page, as checking requires a POST submission
            # Alternatively, check authentication and render index.html if logged in,
            # or auth.html if not logged in. Redirecting to '/' handles both via the index view.
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
    
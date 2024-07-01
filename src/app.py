from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define a route to render the form
@app.route('/')
def home():
    return render_template('home.html')

# Route to handle form submission and CSV upload
@app.route('/train', methods=['POST','GET'])
def train():
    from src.logger import logging
    import src.pipeline.train_pipeline as pipeline
    if request.method == 'POST':
        if 'csv_file' in request.files:
            csv_file = request.files['csv_file']
            
            if csv_file.filename.endswith('.csv'):
                # Save the CSV file
                logging.info('Initating the reading of the file.')
                csv_file.save(csv_file.filename)
                
                # Process the CSV file (example: read it as pandas DataFrame)
                train_data = pd.read_csv(csv_file.filename)
                trainer = pipeline.TrainPipeline()
                _,_,r2_score=trainer.initate_train_model(train_data)
                return f"Model is successfully trained and r2 score is :{r2_score}."
            else:
                return "Only CSV files are allowed."
    else:
        return render_template('train.html')
    

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        from src.logger import logging
        if 'csv_file' in request.files:
            # Handle CSV file upload
            csv_file = request.files['csv_file']
            if csv_file.filename == '':
                return "No selected file"

            if csv_file and allowed_file(csv_file.filename):
                filename = secure_filename(csv_file.filename)
                csv_file.save(os.path.join(filename))
                # Process CSV file (for example, read as DataFrame)
                df = pd.read_csv(os.path.join(filename))
                # Do something with the DataFrame (e.g., perform predictions)
                logging.info('Importing the predict pipeline and initate the prediction process.')
                from src.pipeline.predict_pipeline import PredictPipeline
                predict = PredictPipeline()
                df['result'] = predict.initate_predict(data=df.copy())
                return df.to_html()

        else:
            # Handle form submission
            item_identifier = request.form['item_identifier']
            item_weight = float(request.form['item_weight'])
            item_fat_content = request.form['item_fat_content']
            item_visibility = float(request.form['item_visibility'])
            item_type = request.form['item_type']
            item_mrp = float(request.form['item_mrp'])
            outlet_identifier = request.form['outlet_identifier']
            outlet_establishment_year = int(request.form['outlet_establishment_year'])
            outlet_size = request.form['outlet_size']
            outlet_location_type = request.form['outlet_location_type']
            outlet_type = request.form['outlet_type']

            # Here you can process the form data, for example, print it
            logging.info('Converting data into dataset for prediction.')
            import src.utils as util
            df = util.CustomDataSetTest(
                Item_Identifier =item_identifier,
                Item_Fat_Content = item_fat_content,
                Item_MRP = item_mrp,
                Item_Type = item_type,
                Item_Visibility = item_visibility,
                Item_Weight = item_weight,
                Outlet_Establishment_Year = outlet_establishment_year,
                Outlet_Identifier = outlet_identifier,
                Outlet_Location_Type = outlet_location_type,
                Outlet_Size = outlet_size,
                Outlet_Type = outlet_type,
            )
            logging.info('Importing the predict pipeline and initate the prediction process.')
            import src.pipeline.predict_pipeline as pipeline
            predict = pipeline.PredictPipeline()
            df['result'] = predict.initate_predict(data=df.copy())
            return df.to_html()
    else:
        return render_template('predict.html')

# Function to check allowed file types (if needed)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'


# Run the application
if __name__ == '__main__':
    app.run(debug=True)

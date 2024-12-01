from flask import Flask, request, jsonify
import pickle
import pandas as pd

# local (and relative) path to dataset used for training
DATASET_PATH = "../../data/processed/training_boston.csv"

# load the developed xgboost model
with open("model/xgboost_model.pkl", "rb") as model_file: # rb param for (reading in binary format)
    model = pickle.load(model_file)

app = Flask(__name__)

df = pd.read_csv(DATASET_PATH)

@app.route("/dataset/info", methods=["GET"])
def dataset_info():
    """
    get overall information about the dataset, such as:
    - number of data points
    - feature statistics
    - target feature statistics
    """
    try:
        num_data_points = len(df)
        stats = {}

        for column in df.columns:
            stats[column] = {
                "mean": df[column].mean(),
                "variance": df[column].var(),
                "std": df[column].std(),
                "min val": df[column].min(),
                "max val": df[column].max(),
            }

        return jsonify({
            "num_data_points": num_data_points,
            "statistics": stats
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/dataset/columns", methods=["GET"])
def dataset_columns():
    """
    get the list of features in the dataset.
    """
    try:
        columns = df.columns.tolist()
        return jsonify({"columns": columns}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# predict for single house (single row of data)
@app.route("/predict", methods=["POST"])
def predict_single():
    """
    predict house price for a single row of data
    expects json input with relevant features

    example request body:
    {
        "feature1": val1,
        "feature2": val2,
        ...
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # json to df
        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)
        predicted_price = float(prediction[0])

        # return pred with success stat code
        # remember that prediction model.predict() returns an np array although the result is single scalar val
        return jsonify({"predicted_price": str(round(predicted_price * 1000,2)) + " $"}), 200

    # TODO - handling missing values

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_bulk", methods=["POST"])
def predict_bulk():
    """
    predict house prices for multiple rows of data (bulk)
    expects a CSV file uploaded as 'file' in the form-data.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        input_df = pd.read_csv(file)

        if input_df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400

        predictions = model.predict(input_df)
        # predictions as a list
        return jsonify({"predicted_prices": [str(round(float(pred * 1000),2)) + " $" for pred in predictions.tolist()]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)

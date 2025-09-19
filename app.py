from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.utils import pad_sequences
import json

app = Flask(__name__)

# Global variables for model and data
model = None
go_terms = None
go_definitions = None
char_dict = None

def load_model_and_data():
    """Load the trained model and necessary data"""
    global model, go_terms, go_definitions, char_dict

    # Load the trained model
    model = tf.keras.models.load_model('assets/model.keras')

    # Create amino acid encoding dictionary
    codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    char_dict = {}
    for index, val in enumerate(codes):
        char_dict[val] = index + 1

    # Load GO terms (molecular function categories)
    go_terms = [
        'GO:0090729', 'GO:0005179', 'GO:0016829', 'GO:0046872', 'GO:0008081',
        'GO:0004867', 'GO:0099106', 'GO:0015459', 'GO:0004252', 'GO:0005246',
        'GO:0004623', 'GO:0005509', 'GO:0008200', 'GO:0017080', 'GO:0019834',
        'GO:0008270', 'GO:0030550', 'GO:0004222', 'GO:0052740', 'GO:0052739',
        'GO:0008970', 'GO:0050025', 'GO:0050029', 'GO:0106329', 'GO:0008191',
        'GO:0004620', 'GO:0008233', 'GO:0016504', 'GO:0060422', 'GO:0000166',
        'GO:0106411', 'GO:0000287', 'GO:0008237', 'GO:0005102', 'GO:0005216',
        'GO:0008201', 'GO:0008236', 'GO:0008289', 'GO:0019870', 'GO:0019871',
        'GO:0004465', 'GO:0016491', 'GO:0042802', 'GO:0047498', 'GO:0048018',
        'GO:0001515', 'GO:0001716', 'GO:0042803', 'GO:0003677', 'GO:0050660',
        'GO:0003990', 'GO:0003993', 'GO:0004175', 'GO:0004177', 'GO:0004415',
        'GO:0008239', 'GO:0004556', 'GO:0004860', 'GO:0004866', 'GO:0004869',
        'GO:0005154', 'GO:0005184', 'GO:0005185', 'GO:0005507', 'GO:0005516',
        'GO:0005520', 'GO:0005534', 'GO:0030395', 'GO:0033296', 'GO:0008061',
        'GO:0008083', 'GO:0016603', 'GO:0016787', 'GO:0019855', 'GO:0003676',
        'GO:0016853', 'GO:0017081', 'GO:0044325', 'GO:0030246', 'GO:0030414',
        'GO:0033906', 'GO:0043262', 'GO:0048019', 'GO:0070320', 'GO:0080030',
        'GO:0140628'
    ]

    # Load GO term definitions from JSON file
    try:
        with open('assets/go.json', 'r', encoding='utf-8') as f:
            go_definitions = json.load(f)
    except FileNotFoundError:
        print("Warning: assets/go.json not found. Using empty definitions.")
        go_definitions = {}
    except UnicodeDecodeError:
        print("Warning: Unicode error reading go.json. Using empty definitions.")
        go_definitions = {}

def get_go_info(go_term):
    """Get GO term name and definition with fallback defaults"""
    if go_term in go_definitions:
        return {
            'name': go_definitions[go_term]['name'],
            'definition': go_definitions[go_term]['definition']
        }
    else:
        return {
            'name': f'Unknown function ({go_term})',
            'definition': 'Function not defined in current database.'
        }

def encode_sequence(sequence):
    """Encode amino acid sequence for model input"""
    encoded_list = []
    for code in sequence.upper():
        encoded_list.append(char_dict.get(code, 0))

    # Pad sequence to 600 characters
    padded = pad_sequences([encoded_list], maxlen=600, padding='post', truncating='post')
    return padded

def predict_functions(sequence, top_n=10):
    """Predict molecular functions for given protein sequence"""
    if not model:
        return {"error": "Model not loaded"}

    try:
        # Encode the sequence
        encoded_seq = encode_sequence(sequence)

        # Make prediction
        predictions = model.predict(encoded_seq, verbose=0)

        # Get top N predictions
        if top_n == 'all':
            sorted_indices = np.argsort(predictions[0])[::-1]
        else:
            sorted_indices = np.argsort(predictions[0])[::-1][:top_n]

        results = []
        for i in sorted_indices:
            go_term = go_terms[i]
            go_info = get_go_info(go_term)
            results.append({
                'go_term': go_term,
                'name': go_info['name'],
                'definition': go_info['definition'],
                'confidence': float(predictions[0][i]),
                'confidence_percent': round(float(predictions[0][i]) * 100, 2)
            })

        return {
            'sequence': sequence[:50] + '...' if len(sequence) > 50 else sequence,
            'sequence_length': len(sequence),
            'predictions': results
        }

    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    """Main page with input form"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    data = request.get_json()

    if not data or 'sequence' not in data:
        return jsonify({'error': 'No sequence provided'}), 400

    sequence = data['sequence'].strip()
    top_n = data.get('top_n', 10)

    # Handle 'all' option
    if top_n == 'all':
        top_n = 'all'
    else:
        top_n = int(top_n)

    if not sequence:
        return jsonify({'error': 'Empty sequence provided'}), 400

    # Validate sequence contains only valid amino acids
    valid_acids = set('ACDEFGHIKLMNPQRSTVWY')
    if not all(c.upper() in valid_acids for c in sequence):
        return jsonify({'error': 'Invalid amino acids in sequence. Use only standard 20 amino acids.'}), 400

    result = predict_functions(sequence, top_n)
    return jsonify(result)

if __name__ == '__main__':
    print("Loading model and data...")
    load_model_and_data()
    print("Model loaded successfully!")
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)
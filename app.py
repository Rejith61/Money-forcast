import os
import io
import csv
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/forecast', methods=['POST'])
def forecast():
    try:
        # Check if salary is provided
        if 'salary' not in request.form:
            return jsonify({'error': 'Monthly salary is required'}), 400
        
        try:
            salary = float(request.form['salary'])
            if salary <= 0:
                return jsonify({'error': 'Salary must be a positive number'}), 400
        except ValueError:
            return jsonify({'error': 'Salary must be a valid number'}), 400
        
        # Check if CSV file is provided
        if 'csv_file' not in request.files:
            return jsonify({'error': 'CSV file is required'}), 400
        
        file = request.files['csv_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get forecast months
        try:
            forecast_months = int(request.form.get('forecast_months', 3))
            if forecast_months <= 0:
                return jsonify({'error': 'Forecast months must be a positive number'}), 400
        except ValueError:
            return jsonify({'error': 'Forecast months must be a valid number'}), 400
        
        # Read and validate CSV content
        try:
            csv_content = file.read().decode('utf-8')
            csv_file = io.StringIO(csv_content)
            csv_reader = csv.reader(csv_file)
            
            # Read header
            headers = next(csv_reader)
            if len(headers) != 2 or headers[0].lower() != 'category' or headers[1].lower() != 'amount':
                return jsonify({'error': 'CSV file must have headers: category,amount'}), 400
            
            # Parse data
            current_data = []
            for row in csv_reader:
                if len(row) != 2:
                    continue
                
                category, amount_str = row
                category = category.strip()
                amount_str = amount_str.strip()
                
                if not category:
                    continue
                
                try:
                    amount = float(amount_str)
                except ValueError:
                    return jsonify({'error': f'Invalid amount for category {category}: {amount_str}'}), 400
                
                current_data.append({'category': category, 'amount': amount, 'month': 4})
            
            if not current_data:
                return jsonify({'error': 'No valid data found in CSV file'}), 400
            
            # Extract unique categories
            categories = list(set(item['category'] for item in current_data))
            
            # Create historical data (3 months) for linear regression
            historical_data = []
            for month in range(1, 4):
                for category in categories:
                    # Find current amount for this category
                    current_amount = next(
                        (item['amount'] for item in current_data if item['category'] == category), 
                        0
                    )
                    
                    # Create synthetic historical data with slight variance
                    variance = np.random.uniform(-0.1, 0.1)  # -10% to +10% variance
                    historical_data.append({
                        'category': category,
                        'amount': current_amount * (1 + variance),
                        'month': month
                    })
            
            # Combine historical and current data
            all_data = historical_data + current_data
            
            # Generate forecast
            forecast_data = generate_forecast(all_data, categories, salary, forecast_months)
            
            return jsonify({
                'categories': categories,
                'forecast': forecast_data
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing CSV file: {str(e)}'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

def generate_forecast(data, categories, salary, forecast_months):
    result = []
    
    # Existing months in the data (for charting)
    for month in range(1, 5):
        month_data = {'month': month, 'savings': 0}
        total_expense = 0
        
        for category in categories:
            category_items = [item for item in data if item['category'] == category and item['month'] == month]
            if category_items:
                amount = category_items[0]['amount']
                month_data[category] = amount
                total_expense += amount
        
        month_data['savings'] = salary - total_expense
        result.append(month_data)
    
    # Future months (forecast)
    for month in range(5, 5 + forecast_months):
        month_data = {'month': month, 'savings': 0}
        total_expense = 0
        
        for category in categories:
            # Get data for this category
            category_data = [item for item in data if item['category'] == category]
            
            if category_data:
                # Prepare data for linear regression
                X = np.array([[item['month']] for item in category_data])
                y = np.array([item['amount'] for item in category_data])
                
                # Fit linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict amount for current month
                predicted_amount = max(0, float(model.predict([[month]])[0]))
                month_data[category] = predicted_amount
                total_expense += predicted_amount
        
        month_data['savings'] = salary - total_expense
        result.append(month_data)
    
    return result

if __name__ == '__main__':
    app.run(debug=True)
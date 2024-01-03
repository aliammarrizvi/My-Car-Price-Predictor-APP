import streamlit as st
import pandas as pd
import joblib
import time

def get_parameter_values(parameter):
	if parameter == 'assembly':
		return ["Local", "Imported"]
	elif parameter == 'make':
		return ["Toyota", "Honda", "Suzuki", "Daihatsu", "Mitsubishi", "Hyundai", "KIA", "Nissan", "Other", "Mercedes", "Changan"]
	elif parameter == 'model':
		return ["Corolla", "City", "Yaris", "Swift", "Civic", "Other", "Mehran", "Cultus", "Vitz", "Alto", "Wagon"]
	elif parameter == 'transmission':
		return ["Manual", "Automatic"]
	elif parameter == 'fuel':
		return ["Petrol", "Hybrid", "Diesel", "Other"]
	elif parameter == 'color':
		return ["Other", "Blue", "Super White", "Grey", "Silver", "Black", "Solid White", "White", "Taffeta White", "Attitude Black", "Graphite Grey"]
	elif parameter == 'registered':
		return ["Lahore", "Punjab", "Islamabad", "Un-Registered", "Sindh", "Other", "Karachi", "Faisalabad", "Multan", "Peshawar", "Rawalpindi"]
	elif parameter == 'engine':
		return [660,1000,1300,1600]
	elif parameter == 'mileage':
		return [50000,100000,150000,200000]
	elif parameter == 'year':
		return [2009,2011,2013,2015,2017,2019,2021,2023]
	elif parameter == 'body':
		return ['Sedan', 'Hatchback', 'SUV', 'Other', 'Unknown', 'Crossover']
	elif parameter == 'city':
		return ['Peshawar', 'Lahore', 'Other', 'Islamabad', 'Karachi', 'Rawalpindi']
	else:
		return []  # Default case, you may adjust it based on your requirements

# Data for select boxes
parameter_list = ['city', 'assembly', 'body', 'make', 'model', 'year', 'engine', 'transmission', 'fuel', 'color', 'registered', 'mileage']
parameter_input_values = []
parameter_description = ['City', 'Assembly', 'Body', 'Make', 'Model', 'Year', 'Engine', 'Transmission', 'Fuel', 'Color', 'Registered', 'Mileage']
parameter_default_values = ['Lahore', 'Local', 'Crossover', 'Toyota', 'Corolla', '2019', '660', 'Manual', 'Petrol', 'Silver', 'Lahore', '150000']

model_path = r''

with st.spinner('Fetching Latest ML Model'):
	# Use joblib to load in the pre-trained model
	encode_model_imported=joblib.load(model_path+'ENCODER_MODEL.pkl')
	pred_model_imported=joblib.load(model_path+'MY_CAR_PRICE.pkl')
	time.sleep(1)
	st.success('WELCOME TO THE CAR PRICE PREDICTION MODEL!')

st.title('Car Price Prediction App \n\n')

for parameter, parameter_df, parameter_desc in zip(parameter_list, parameter_default_values, parameter_description):
	st.subheader(f'Select value for {parameter_desc}')
	parameter_values = get_parameter_values(parameter)
	parameter_input_values.append(st.selectbox(key=parameter,label=parameter_desc, options=parameter_values))


st.write('\n','\n')
st.title('Your Input Summary')
st.write(parameter_input_values)
st.write('\n','\n')

def predict(input_predict, feature_names):


	input_variables = pd.DataFrame([input_predict['data']], columns=feature_names, index=['input'])
	
	# st.write(input_variables)

	numeric_columns = input_variables.select_dtypes(include=['int64', 'float64']).columns.to_list()

	# numeric_columns.remove('price')

	categorical_columns = input_variables.select_dtypes(include=['object']).columns

	my_predict_value=input_variables
 
	my_predict_value_cat=my_predict_value[categorical_columns]
	
	my_predict_value_num=my_predict_value[numeric_columns]

	my_predict_value_cat_e=encode_model_imported.transform(my_predict_value_cat)

	my_predict_cat_e_df=pd.DataFrame.sparse.from_spmatrix(data=my_predict_value_cat_e[0:,0:],columns=encode_model_imported.get_feature_names_out(), index=my_predict_value.index)

	consolidated_df_pred = pd.concat([my_predict_value_num, my_predict_cat_e_df], axis=1)

	prediction=pred_model_imported.predict(consolidated_df_pred)


	# Get the model's prediction
	# prediction = model.predict(input_variables)

	# print("Predicted Price of your desired vehicle is: ", prediction)

	ret = {"prediction": float(prediction)}

	return ret

if st.button("Click Here to Predict"):
	PARAMS = {'data': parameter_input_values}
	r = predict(PARAMS, parameter_list)

	st.write('\n','\n')
	prediction_value = r.get('prediction')

	st.write(f'Your Car Price is: Rs. **{prediction_value}**')

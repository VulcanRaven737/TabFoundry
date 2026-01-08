import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import datetime

file = pd.read_csv('cleaned_aadhaar_dataset.csv')

aadhaar_df = file.copy()

# Find all unique state names

#unique_states = aadhaar_df['state'].unique()
unique_states = aadhaar_df['district'].unique()
print("States before cleaning:")
for state in unique_states:
    print(state)

# Example mapping for cleaning state names
state_mapping = {
    'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands',
    'Dadra & Nagar Haveli' : 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra and Nagar Haveli' : 'Dadra and Nagar Haveli and Daman and Diu',
    'Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Daman and Diu' : 'Dadra and Nagar Haveli and Daman and Diu',
    'The Dadra And Nagar Haveli And Daman And Diu' : 'Dadra and Nagar Haveli and Daman and Diu',
    'Jammu & Kashmir' : 'Jammu and Kashmir',
    'Jammu And Kashmir' : 'Jammu and Kashmir',
    'Orissa' : 'Odisha',
    'Pondicherry' : 'Puducherry',

}

district_mapping = {
    # Andaman and Nicobar Islands
    'Nicobars': 'Nicobar',

    # Andhra Pradesh
    'Ananthapur': 'Anantapur',
    'Ananthapuramu': 'Anantapur',
    'Cuddapah': 'Y. S. R',
    'Karim Nagar': 'Karimnagar',
    'K.v. Rangareddy': 'Ranga Reddy',
    'K.V.Rangareddy': 'Ranga Reddy',
    'Mahabub Nagar': 'Mahabubnagar',
    'Mahbubnagar': 'Mahabubnagar',
    'Nellore': 'Sri Potti Sriramulu Nellore',
    'Rangareddi': 'Ranga Reddy',
    'Spsr Nellore': 'Sri Potti Sriramulu Nellore',
    'Visakhapatanam': 'Visakhapatnam',

    # Assam
    'North Cachar Hills': 'Dima Hasao',
    'Sibsagar': 'Sivasagar',
    'Tamulpur District': 'Tamulpur',

    # Bihar
    'Aurangabad(bh)': 'Aurangabad',
    'Bhabua': 'Kaimur (Bhabua)',
    'East Champaran': 'Purbi Champaran',
    'Monghyr': 'Munger',
    'Pashchim Champaran': 'West Champaran',
    'Purba Champaran': 'Purbi Champaran',
    'Purnea': 'Purnia',
    'Samstipur': 'Samastipur',
    'Sheikpura': 'Sheikhpura',

    # Chhattisgarh
    'Dakshin Bastar Dantewada': 'Dantewada',
    'Gaurella Pendra Marwahi': 'Gaurela-pendra-marwahi',
    'Janjgir - Champa': 'Janjgir Champa',
    'Janjgir-champa': 'Janjgir Champa',
    'Kawardha': 'Kabeerdham',
    'Mohla-Manpur-Ambagarh Chouki': 'Mohalla-Manpur-Ambagarh Chowki',
    'Uttar Bastar Kanker': 'Kanker',

    # Dadra and Nagar Haveli and Daman and Diu
    'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli',
    'Dadra And Nagar Haveli': 'Dadra and Nagar Haveli',

    # Delhi
    'Najafgarh': 'South West Delhi',
    'North East': 'North East Delhi',
    'North East   *': 'North East Delhi',

    # Goa
    'Bardez': 'North Goa',

    # Gujarat
    'Ahmadabad': 'Ahmedabad',
    'Banas Kantha': 'Banaskantha',
    'Dohad': 'Dahod',
    'Panchmahals': 'Panch Mahals',
    'Sabar Kantha': 'Sabarkantha',
    'Surendra Nagar': 'Surendranagar',
    'The Dangs': 'Dang',

    # Haryana
    'Gurgaon': 'Gurugram',
    'Jhajjar *': 'Jhajjar',
    'Mewat': 'Nuh',
    'Yamuna Nagar': 'Yamunanagar',

    # Himachal Pradesh
    'Lahul & Spiti': 'Lahaul and Spiti',
    'Lahul and Spiti': 'Lahaul and Spiti',

    # Jammu and Kashmir
    'Badgam': 'Budgam',
    'Bandipur': 'Bandipore',
    'Shupiyan': 'Shopian',

    # Jharkhand
    'East Singhbum': 'East Singhbhum',
    'Garhwa *': 'Garhwa',
    'Hazaribag': 'Hazaribagh',
    'Kodarma': 'Koderma',
    'Pakur': 'Pakaur',
    'Palamau': 'Palamu',
    'Pashchimi Singhbhum': 'West Singhbhum',
    'Purbi Singhbhum': 'East Singhbhum',
    'Sahebganj': 'Sahibganj',
    'Seraikela-kharsawan': 'Seraikela-Kharsawan',

    # Karnataka
    'Bagalkot *': 'Bagalkot',
    'Bangalore': 'Bengaluru Urban',
    'Bangalore Rural': 'Bengaluru Rural',
    'Belgaum': 'Belagavi',
    'Bellary': 'Ballari',
    'Bengaluru': 'Bengaluru Urban',
    'Bengaluru South': 'Bengaluru Urban',
    'Chamarajanagar *': 'Chamarajanagar',
    'Chamrajanagar': 'Chamarajanagar',
    'Chamrajnagar': 'Chamarajanagar',
    'Chickmagalur': 'Chikkamagaluru',
    'Chikmagalur': 'Chikkamagaluru',
    'Davanagere': 'Davangere',
    'Gadag *': 'Gadag',
    'Gulbarga': 'Kalaburagi',
    'Hasan': 'Hassan',
    'Haveri *': 'Haveri',
    'Mysore': 'Mysuru',
    'Ramanagar': 'Ramanagara',
    'Shimoga': 'Shivamogga',
    'Tumkur': 'Tumakuru',
    'yadgir': 'Yadgir',

    # Kerala
    'Kasargod': 'Kasaragod',

    # Madhya Pradesh
    'Ashok Nagar': 'Ashoknagar',
    'East Nimar': 'Khandwa',
    'Harda *': 'Harda',
    'Hoshangabad': 'Narmadapuram',
    'Khargone': 'West Nimar',
    'Narsinghpur': 'Narsimhapur',

    # Maharashtra
    'Ahmadnagar': 'Ahmednagar',
    'Ahmed Nagar': 'Ahmednagar',
    'Bid': 'Beed',
    'Buldana': 'Buldhana',
    'Chatrapati Sambhaji Nagar': 'Chhatrapati Sambhajinagar',
    'Gondia': 'Gondiya',
    'Gondiya *': 'Gondiya',
    'Mumbai( Sub Urban )': 'Mumbai Suburban',
    'Nandurbar *': 'Nandurbar',
    'Osmanabad': 'Dharashiv',
    'Raigarh(MH)': 'Raigad',
    'Washim *': 'Washim',

    # Meghalaya
    'East Khasi Hills' : 'Khasi Hills',
    'West Khasi Hills': 'Khasi Hills',
    'Eastern West Khasi Hills': 'Khasi Hills',
    'West Jaintia Hills': 'Jaintia Hills',
    'East Jaintia Hills': 'Jaintia Hills',
    'South West Garo Hills': 'Garo Hills',
    'North Garo Hills': 'Garo Hills',
    'South Garo Hills': 'Garo Hills',
    'West Garo Hills': 'Garo Hills',
    'East Garo Hills': 'Garo Hills',
    'South West Khasi Hills': 'Khasi Hills',
    
    # Mizoram
    'Mammit': 'Mamit',

    # Nagaland
    'Meluri': 'Phek',

    # Odisha
    'ANUGUL': 'Angul',
    'Anugul': 'Angul',
    'Baleswar': 'Baleshwar',
    'Boudh': 'Baudh',
    'Jagatsinghapur': 'Jagatsinghpur',
    'jajpur': 'Jajpur',
    'Jajapur': 'Jajpur',
    'JAJPUR': 'Jajpur',
    'Khorda': 'Khordha',
    'Nabarangapur': 'Nabarangpur',
    'NUAPADA': 'Nuapada',
    'Sonapur': 'Subarnapur',
    'Sundergarh': 'Sundargarh',

    # Puducherry
    'Pondicherry': 'Puducherry',

    # Punjab
    'Firozpur': 'Ferozepur',
    'Nawanshahr': 'Shaheed Bhagat Singh Nagar',
    'S.A.S Nagar': 'SAS Nagar (Mohali)',
    'S.A.S Nagar(Mohali)': 'SAS Nagar (Mohali)',
    'Sri Muktsar Sahib': 'Muktsar',
    
    # Rajasthan
    'Chittaurgarh': 'Chittorgarh',
    'Dholpur': 'Dhaulpur',
    'Jalor': 'Jalore',
    'Jhunjhunun': 'Jhunjhunu',

    # Sikkim
    'East': 'East Sikkim',
    'Namchi': 'South Sikkim',
    'North': 'North Sikkim',
    'South': 'South Sikkim',
    'West': 'West Sikkim',

    # Tamil Nadu
    'Kanniyakumari': 'Kanyakumari',
    'Kancheepuram': 'Kanchipuram',
    'Thiruvallur': 'Tiruvallur',
    'Thiruvarur': 'Tiruvarur',
    'Tirupathur': 'Tirupattur',
    'Viluppuram': 'Villupuram',

    # Telangana
    'Jangoan': 'Jangaon',
    'Medchal−malkajgiri': 'Medchal-malkajgiri',
    'Warangal (urban)': 'Hanumakonda',
    'Warangal Urban': 'Hanumakonda',
    'Warangal Rural': 'Warangal',
    'Yadadri.': 'Yadadri Bhuvanagiri',

    # Tripura
    'Dhalai  *': 'Dhalai',

    # Uttar Pradesh
    'Allahabad': 'Prayagraj',
    'Bara Banki': 'Barabanki',
    'Bulandshahar': 'Bulandshahr',
    'Faizabad': 'Ayodhya',
    'Jyotiba Phule Nagar': 'Amroha',
    'Mahrajganj': 'Maharajganj',
    'Sant Ravidas Nagar': 'Bhadohi',
    'Sant Ravidas Nagar Bhadohi': 'Bhadohi',
    'Shravasti': 'Shrawasti',
    'Siddharthnagar': 'Siddharth Nagar',

    # Uttarakhand
    'Garhwal': 'Pauri Garhwal',
    'Hardwar': 'Haridwar',

    # West Bengal
    '24 Paraganas North': 'North 24 Parganas',
    '24 Paraganas South': 'South 24 Parganas',
    'Barddhaman': 'Bardhaman',
    'Coochbehar': 'Cooch Behar',
    'Darjiling': 'Darjeeling',
    'Dinajpur Uttar': 'Uttar Dinajpur',
    'East Midnapore': 'Purba Medinipur',
    'Haora': 'Howrah',
    'hooghly': 'Hooghly',
    'Hugli': 'Hooghly',
    'Koch Bihar': 'Cooch Behar',
    'Malda': 'MALDA',
    'Maldah': 'MALDA',
    'MALDA': 'Malda',
    'Medinipur': 'Purba Medinipur',
    'Medinipur West': 'Paschim Medinipur',
    'nadia': 'Nadia',
    'NADIA': 'Nadia',
    'North Dinajpur': 'Uttar Dinajpur',
    'North Twenty Four Parganas': 'North 24 Parganas',
    'Puruliya': 'Purulia',
    'South Dinajpur': 'Dakshin Dinajpur',
    'South Twenty Four Parganas': 'South 24 Parganas',
    'West Midnapore': 'Paschim Medinipur'
}

# Clean up state names
aadhaar_df['district'] = aadhaar_df['district'].replace(district_mapping)
df_sorted = aadhaar_df.sort_values(['date']).reset_index(drop=True)

# Check unique states after cleaning
cleaned_states = aadhaar_df['district'].unique()
print("\nStates after cleaning:")
for state in cleaned_states:
    print(state)

output_path = 'cleaned_aadhaar_dataset.csv'
aadhaar_df.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to {output_path}")
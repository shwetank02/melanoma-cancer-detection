import tensorflow as tf
import streamlit as st
from IPython.display import HTML
import base64
import numpy as np
import pandas as pd

model = tf.keras.models.load_model(r"C:\Users\SHWETANK\Downloads\resnet50model.hdf5")
st.write("""
         # Melanoma Detection
         """
         )
st.write("This is a model to detect the presence of malenoma(skin cancer)")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


import cv2
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)) / 255.

    img_reshape = img_resize[np.newaxis, ...]

    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    if np.argmax(prediction) == 0:
        st.write("Benign")
    elif np.argmax(prediction) == 1:
        st.write("Malignant, but don't panic. You can get help from the below mentioned sources")

    st.write(prediction)
    if(np.argmax(prediction)==0):
        st.text('Congratulations.....You are SAFE 😊')
    else:
        hos = {}
        hos['Andhra Pradesh'] = [['Apollo Hospitals Enterprise Limited',
                                  'Apollo Hopsitals, Health City, Arilova, Chinagadhili, Visakhapatnam 530040, AP',
                                  'https://www.apollohospitals.com/'],
                                 ['Homi Bhabha Cancer Hospital & Research Centre',
                                  'Aganampudi, Gajuwaka Mandalam, Visakhapatnam 530 053, Andhra Pradesh',
                                  'https://tmc.gov.in']]
        hos['Assam'] = [['Dr. B. Borooah Cancer Institute', 'Gopinath Nagar, Guwahati, Assam 781 016',
                         'http://www.bbcionline.org/contactus.htm']]
        hos['Bihar'] = [['Indira Gandhi Institute of Medical Sciences, Patna', 'Sheikhpura, Patna 800014, Bihar',
                         'http://www.igims.org/'],
                        ['Mahavir Cancer Sansthan', 'Khagoul Road, Phulwari Sharif, Patna, Bihar 801 505',
                         'http://www.mahavircancersansthan.com/']]
        hos['Chandigarh'] = [
            ['Postgraduate Institute of Medical Education & Research (PGI Chandigarh)- Regional Cancer Centre',
             'Sector 12, Chandigarh 160 012', 'http://pgimer.edu.in/'],
            ]
        hos['Chattisgarh'] = [
            ['Regional Cancer Centre, Raipur', 'Raipur, Chattisgarh', 'http://www.ptjnmcraipur.in/Radiotherapy.htm'],
            ['BALCO MEDICAL CENTRE (Vedanta Medical Research Foundation)',
             'Road, Sector 36, Uparwara, Naya Raipur, Chhattisgarh 493661', 'http://www.balcomedicalcentre.com/']]
        hos['Delhi'] = [['All India Institute of Medical Sciences, Dr. B. R. A. Institute Rotary Cancer Hospital',
                         'Ansari Nagar, New Delhi 110 029', 'http://www.aiims.edu/en.html'],
                        ['Delhi State Cancer Institute', 'EAST: Dilshad Garden, Delhi 110 095',
                         '	http://www.dsci.nic.in/home.html'],
                        ['CanKids', 'D 7/7, Vasant Vihar New Delhi', 'www.cankidsindia.org'],
                        ['CanSupport Services',
                         'A2 (2nd Floor), Gulmohar Park, (Opp Kamla Nehru College) New Delhi-110049',
                         'https://cansupport.org/'],
                        ['Indian Cancer Society, Delhi',
                         'B 63-64,Basement, South Extension Part-II, New Delhi - 110 049',
                         'http://indiancancersocietydelhi.in/']]
        hos['Goa'] = [['Goa Medical College', 'Bambolim Goa', 'http://www.gmc.goa.gov.in/index.php/en/']]
        hos['Maharashtra'] = [
            ['Aditya Birla memorial hospital', 'Aditya Birla Hospital Marg, P. O. Chinchwad, Pune 411 033',
             'http://www.adityabirlahospital.com'],
            ['	Apollo Hospital', 'Plot no.13,Parsik hill road,Off Uran road.Sector-23CBD Belapur,NaviMumbai-400614',
             'https://mumbai.apollohospitals.com'],
            ['Apple Saraswati Multispeciality Hospital',
             '804/2, 805/2 circuit house Kadamwadi Road, Bhosalewadi, Kolhapur 416003',
             'http://www.applesaraswati.com/']]

        z=st.selectbox('Please select your state',
                     ('Andhra Pradesh','Assam','Bihar','Chandigarh','Chattisgarh','Delhi',
                      'Goa','Maharashtra'))

        def make_clickable(link):
            # target _blank to open new window
            # extract clickable text to display for your link
            text = link.split('=')[0]
            return f'<a target="_blank" href="{link}">{text}</a>'
        # Randomly fill a dataframe and cache it
        @st.cache(allow_output_mutation=True)
        def get_dataframe():
            df=pd.DataFrame(index=None)
            df['Name']=[i[0] for i in hos[z]]
            df['Address']=[i[1] for i in hos[z]]
            df['Website']=[i[2] for i in hos[z]]
            df['Website'] = df['Website'].apply(make_clickable)
            return df

        df = get_dataframe()

        # link is the column with hyperlinks
        #df['Website'] = df['Website'].apply(make_clickable)
        df = df.to_html(escape=False)
        st.write(df, unsafe_allow_html=True)



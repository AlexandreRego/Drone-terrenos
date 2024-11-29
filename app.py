# import streamlit as st
# import gdown
# import tensorflow as tf
# import io
# from PIL import Image
# import numpy as np
# import pandas as pd
# import plotly.express as px
# import os

# @st.cache_resource
# def carrega_modelo():
#     url = 'https://drive.google.com/uc?id=1xXDRe_gE7AlRKJyxstaLX2bXevi2yT0h'
#     gdown.download(url, 'modelo_quantizado16bits.tflite')
#     interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
#     interpreter.allocate_tensors()

#     return interpreter

# # def carrega_imagem():
# #     uploaded_file = st.file_uploader('Arraste e solte uma imagem aqui ou clique para selecionar uma', type=['png', 'jpg', 'jpeg'])

# #     if uploaded_file is not None:
# #         image_data = uploaded_file.read()
# #         image = Image.open(io.BytesIO(image_data))

# #         st.image(image)
# #         st.success('Imagem foi carregada com sucesso')

# #         image = np.array(image, dtype=np.float32)
# #         image = image / 255.0
# #         image = np.expand_dims(image, axis=0)

# #         return image



# def previsao(interpreter,image):

#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
    
#     interpreter.set_tensor(input_details[0]['index'], image) 
    
#     interpreter.invoke()

#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     classes = ['bom', 'Erosão', 'represas']

#     df = pd.DataFrame()
#     df['classes'] = classes
#     df['probabilidades (%)'] = 100*output_data[0]

#     fig = px.bar(df,y='classes',x='probabilidades (%)',  orientation='h', text='probabilidades (%)', title='Probabilidade de anomalias no terreno')
#     st.plotly_chart(fig)

# def main():
#     st.set_page_config(
#         page_title="Classifica terrenos",
#         # page_icon=" ",
#     )
#     # st.write("# Classifica terrenos!")
    
#     interpreter = carrega_modelo()
    
#     image = carrega_imagem()
    
#     if image is not None:
#         previsao(interpreter, image)



# if __name__ == "__main__":
#     main()

import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path

@st.cache_resource
def carrega_modelo():
    #https://drive.google.com/file/d/1XFoV-l8-hdqc6PVFRLv66__37XxA88Qp/view?usp=drive_link
    url = 'https://drive.google.com/uc?id=1XFoV-l8-hdqc6PVFRLv66__37XxA88Qp'
    gdown.download(url, 'modelo_quantizado16bits.tflite', quiet=False)
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()
    st.success("Modelo carregado com sucesso!")
    return interpreter

def carrega_imagem(interpreter):
    uploaded_file = st.file_uploader('Arraste e solte uma imagem aqui ou clique para selecionar uma', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))
        st.image(image, caption="Imagem carregada com sucesso", use_column_width=True)

        # Verificar dimensões esperadas pelo modelo
        input_shape = interpreter.get_input_details()[0]['shape']
        st.write(f"Dimensões esperadas pelo modelo: {input_shape}")
        image = image.resize((input_shape[1], input_shape[2]))  # Redimensionar para o tamanho esperado
        image = np.array(image, dtype=np.float32) / 255.0  # Normalizar para [0, 1]
        image = np.expand_dims(image, axis=0)  # Adicionar batch dimension

        st.write(f"Dimensões da imagem processada: {image.shape}")
        st.write(f"Tipo de dados da imagem: {image.dtype}")
        return image

    return None

def previsao(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Garantir que o tipo de dado da imagem seja compatível com o modelo
    expected_dtype = input_details[0]['dtype']
    if image.dtype != expected_dtype:
        st.warning(f"Convertendo tipo de dados da imagem de {image.dtype} para {expected_dtype}.")
        image = image.astype(expected_dtype)

    # Fazer a previsão
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Classes do modelo
    classes = ['erosao', 'viavel']
    df = pd.DataFrame({'classes': classes, 'probabilidades (%)': 100 * output_data[0]})
    
    # Plotar gráfico de barras
    fig = px.bar(
        df, y='classes', x='probabilidades (%)',
        orientation='h', text='probabilidades (%)',
        title='Probabilidade de anomalias no terreno'
    )
    st.plotly_chart(fig)

def main():
    st.set_page_config(
        page_title="Classifica Terrenos",
        page_icon="🌍",
    )
    st.title("Classifica Terrenos")
    st.subheader("Carregue uma imagem para análise.")

    # Carregar o modelo
    interpreter = carrega_modelo()

    # Carregar a imagem
    image = carrega_imagem(interpreter)

    # Realizar a previsão
    if image is not None:
        st.write("Realizando previsão...")
        previsao(interpreter, image)

if __name__ == "__main__":
    main()

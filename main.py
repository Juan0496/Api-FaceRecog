from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
from keras.preprocessing import image


#CARGA LA MODELOS 
path_json="./sospechosos.json"
path_h5="./sospechosos.h5"
json_file=open(path_json,"r")
modelo_json=json_file.read()
json_file.close()
modelo=keras.models.model_from_json(modelo_json)
modelo.load_weights(path_h5)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

############################################################################

@app.get("/")
def init():       
    return {"hello"}
############################################################################
@app.post("/recibir/")
async def send_message(file: UploadFile = File(...)):
    #RECIBE IMAGEN
    contents = await file.read()   
    nparr = np.frombuffer(contents, np.uint8)
    #GUARDA LA IMAGEN EN EL PATH
    output_path = "output_image.jpg"
    with open(output_path, "wb") as f:
        f.write(nparr )
    #CARGA LA IMAGEN PARA EL ANALISIS 
    path_img="./output_image.jpg"    
    img = image.load_img(path_img, target_size=(250, 250))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    imagen=np.vstack([x])
    #OBTIENE EL RESULTADO Y LO ALMACEN EN resp
    resp = " "
    y_pred=modelo.predict(imagen)
    print(y_pred[0])
    for i in y_pred[0]:
        print(i)
        if (i != 0 and i!= 1):                 
            return {"respuesta": "Imagen incorrecta, intente de nuevo"}
       
    resultados=["Su rostro esta con un casco","Su rostro esta descubierto","Su rostro esta con una mascarilla"]    
    for i,v in enumerate(resultados):
        if (y_pred[0][i]*100 == 100 ):          
            resp = v 
    return {"respuesta": resp}
       
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
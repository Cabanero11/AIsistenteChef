import datetime
import json
import os
import streamlit as st
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
HISTORIAL_FILE = 'historial_cocina.json'

st.set_page_config(page_title='Chef Rapidín V2', page_icon='🍳', layout='centered')

# -----------------------------------------------------------------------------
# CARGA DE RECURSOS (Modelo + RAG + Dataset)
# -----------------------------------------------------------------------------
@st.cache_resource
def cargar_recursos():
    model_id = 'Qwen/Qwen2.5-3B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map='auto',
        trust_remote_code=True
    )
    
    retriever = SentenceTransformer('all-MiniLM-L6-v2')
    return tokenizer, model, retriever

@st.cache_data
def cargar_y_preparar_dataset():
    ds = load_dataset('datahiveai/recipes-with-nutrition', split='train')
    df = pd.DataFrame(ds).head(8000)
    
    mapeo_columnas = {
        'recipe_name': 'title',
        'ingredient_lines': 'instructions',
        'ingredients': 'ingredients',
        'health_labels': 'health_labels',
        'dish_type': 'dish_type',
        'meal_type': 'meal_type'
    }
    
    df = df.rename(columns=mapeo_columnas)
    columnas_finales = ['title', 'ingredients', 'health_labels', 'dish_type', 'meal_type', 'instructions']
    
    for col in columnas_finales:
        if col not in df.columns:
            df[col] = ''
        else:
            df[col] = df[col].fillna('')
            
    return df[columnas_finales]

# Inicialización segura y secuencial de las variables globales
try:
    tokenizer, model, retriever = cargar_recursos()
    df_recetas = cargar_y_preparar_dataset()
except Exception as e:
    st.error(f'❌ Error de inicialización: {e}')
    st.stop()

# -----------------------------------------------------------------------------
# LÓGICA RAG OPTIMIZADA
# -----------------------------------------------------------------------------
@st.cache_resource
def generar_embeddings_globales(_df_origen):
    # Usamos las columnas mapeadas finales ('title', etc.)
    recetas_text = [
        f"Nombre: {row['title']} | Ingredientes: {row['ingredients']} | Dieta: {row['health_labels']} | Tipo: {row['dish_type']} {row['meal_type']}"
        for _, row in _df_origen.iterrows()
    ]
    return retriever.encode(recetas_text, convert_to_numpy=True)

# Creamos los embeddings pasando el DataFrame ya inicializado de forma segura
embeddings_recetas = generar_embeddings_globales(df_recetas)

def obtener_contexto(query, df_filtrado, embeddings_filtrados):
    if df_filtrado.empty:
        return 'No se encontraron recetas que cumplan con los filtros de dieta o alergias.'
        
    embedding_query = retriever.encode([query], convert_to_numpy=True)
    similitudes = cosine_similarity(embedding_query, embeddings_filtrados)
    
    top_k = min(2, len(df_filtrado))
    indices = np.argsort(similitudes[0])[-top_k:]
    
    # Corregido aquí para usar las columnas unificadas ('title' e 'instructions')
    contexto = '\n'.join([
        f"- {df_filtrado.iloc[i]['title']} (Tipo: {df_filtrado.iloc[i]['meal_type']}/{df_filtrado.iloc[i]['dish_type']}): {df_filtrado.iloc[i]['instructions']}"
        for i in indices
    ])
    return contexto

# -----------------------------------------------------------------------------
# HISTORIAL
# -----------------------------------------------------------------------------
def cargar_historial():
    if not os.path.exists(HISTORIAL_FILE): return []
    try:
        with open(HISTORIAL_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except: return []

def guardar_en_historial(plato, ingredientes, tecnica):
    historial = cargar_historial()
    historial.append({
        'fecha': str(datetime.date.today()),
        'plato': plato, 'ingredientes': ingredientes, 'tecnica': tecnica
    })
    with open(HISTORIAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(historial[-5:], f, ensure_ascii=False, indent=4)

# -----------------------------------------------------------------------------
# INTERFAZ DE USUARIO
# -----------------------------------------------------------------------------
st.title('🍳 Chef Rapidín V2')
historial_previo = cargar_historial()

st.sidebar.header('⚙️ Filtros de Salud y Dieta')

alergias = st.sidebar.text_input('Alergias / Excluir (ej: Peanut, Milk, Gluten)', value='')
tipo_dieta = st.sidebar.multiselect(
    'Tipo de dieta',
    options=['VEGETARIAN', 'VEGAN', 'PALEO', 'KETO_FRIENDLY', 'GLUTEN_FREE', 'DAIRY_FREE'],
    default=[]
)

momento_dia = st.sidebar.selectbox(
    '¿Para cuándo es?',
    options=['Cualquiera', 'lunch', 'dinner', 'breakfast', 'snack'],
    index=0
)

ingredientes_usuario = st.text_input('¿Qué tienes por ahí tirado?', value='rice, tuna, onion')

if st.button('⚡ Generar recetas'):
    df_filtrado = df_recetas.copy()
    indices_validos = np.arange(len(df_recetas))
    
    # --- COPIA LIMPIA DE INGREDIENTES DEL USUARIO ---
    ingredientes_ia = ingredientes_usuario
    
    # 2. Filtro de Tipo de Dieta (Inclusión) + Limpieza de ingredientes en conflicto
    if tipo_dieta:
        for dieta in tipo_dieta:
            mask = df_filtrado['health_labels'].str.upper().str.contains(dieta)
            df_filtrado = df_filtrado[mask]
            indices_validos = indices_validos[mask]
        
        # Si el usuario quiere algo Vegano o Vegetariano, purgamos la carne/pescado del input
        dietas_strings = [d.upper() for d in tipo_dieta]
        if 'VEGAN' in dietas_strings or 'VEGETARIAN' in dietas_strings:
            # Lista de cosas que NO pueden ir bajo ningún concepto
            prohibidos = ['tuna', 'chicken', 'beef', 'pork', 'salmon', 'fish', 'meat', 'ham', 'jamon']
            if 'VEGAN' in dietas_strings:
                # Si es vegano estricto, sumamos lácteos, huevos y quesos
                prohibidos += ['cheese', 'egg', 'milk', 'butter', 'queso', 'huevo', 'yogurt']
            
            # Filtramos la cadena de texto que va a leer el modelo
            palabras_usuario = [p.strip() for p in ingredientes_usuario.split(',')]
            palabras_limpias = [p for p in palabras_usuario if p.lower() not in prohibidos]
            ingredientes_ia = ', '.join(palabras_limpias)
            
    # 3. Filtro de Momento del Día
    if momento_dia != 'Cualquiera':
        mask = df_filtrado['meal_type'].str.lower().str.contains(momento_dia) | df_filtrado['dish_type'].str.lower().str.contains(momento_dia)
        df_filtrado = df_filtrado[mask]
        indices_validos = indices_validos[mask]

    embeddings_filtrados = embeddings_recetas[indices_validos]
    contexto_rag = obtener_contexto(ingredientes_usuario, df_filtrado, embeddings_filtrados)
    texto_historial = ', '.join([f"{h['plato']} ({h['tecnica']})" for h in historial_previo])

    system_prompt = f"""
    Eres un Chef experto. Usa el contexto de recetas reales proporcionado para crear recetas creativas.
    
    Recetas de referencia (USA ESTAS COMO BASE):
    {contexto_rag}
    
    Reglas estrictas:
    - Si las recetas de referencia usan ingredientes que tengo, úsalas.
    - REGLA DE SALUD CRÍTICA: No utilices bajo ningún concepto ingredientes prohibidos por el usuario. Alergias a evitar: {alergias}. Dieta obligatoria: {', '.join(tipo_dieta)}.
    - Respuesta SOLO en JSON. Sin texto adicional ni bloques de código markdown extraños.
    - Humor sarcástico e irónico fuerte.
    - Asegúrate de que las acciones culinarias en español tengan sentido (usa verbos como picar, trocear, batir, nunca inventos raros).
    - Formato JSON con claves: opcion_1, opcion_2, opcion_3. Cada una con nombre, tecnica, tiempo, pasos (lista).
    """

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f'Ingredientes disponibles: {ingredientes_ia}. Momento del día deseado: {momento_dia}.'}
    ]

    # Ahora a ver si parsea el JSON BIEN
    with st.spinner('🧠 Cocinando con RAG y filtrado inteligente...'):
        # Generamos los inputs y obtenemos explícitamente la attention_mask
        inputs = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors='pt'
        ).to(model.device)
        
        # Creamos la máscara de atención para que el modelo no se confunda con el pad_token
        attention_mask = (inputs != tokenizer.pad_token_id).long().to(model.device)
        
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,  # Pasamos la máscara aquí
            max_new_tokens=1000,            # Le damos un pelín más de margen para recetas largas
            temperature=0.3,                # Bajamos la temperatura para que sea más preciso con el JSON
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        texto_generado = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Limpieza y parseo ultra robusto de JSON
        try:
            # Quitamos posibles bloques de markdown que meta el modelo por su cuenta
            texto_limpio = texto_generado.strip()
            if texto_limpio.startswith('```json'):
                texto_limpio = texto_limpio[7:]
            if texto_limpio.startswith('```'):
                texto_limpio = texto_limpio[3:]
            if texto_limpio.endswith('```'):
                texto_limpio = texto_limpio[:-3]
            texto_limpio = texto_limpio.strip()

            inicio, fin = texto_limpio.find('{'), texto_limpio.rfind('}') + 1
            recetas = json.loads(texto_limpio[inicio:fin])
            
            st.session_state['recetas_generadas'] = recetas
            st.session_state['ingredientes_usados'] = ingredientes_usuario
        except Exception as json_err:
            st.error('Error al parsear el JSON de la receta.')
            # Te muestra en la web qué ha respondido el modelo exactamente para poder auditarlo
            with st.expander('🔍 Ver respuesta en bruto del Chef'):
                st.code(texto_generado, language='json')

# -----------------------------------------------------------------------------
# RENDERIZADO
# -----------------------------------------------------------------------------
if 'recetas_generadas' in st.session_state:
    recetas = st.session_state['recetas_generadas']
    for i, (clave, opc) in enumerate(recetas.items()):
        with st.expander(f"🍽️ {opc['nombre']}", expanded=True):
            st.write(f"**Técnica:** {opc['tecnica']} | **Tiempo:** {opc['tiempo']}")
            for paso in opc['pasos']: 
                st.write(f"• {paso}")
            if st.button(f"Guardar {opc['nombre']}", key=f'btn_{i}'):
                guardar_en_historial(opc['nombre'], st.session_state['ingredientes_usados'], opc['tecnica'])
                st.success('¡Guardado en el historial!')
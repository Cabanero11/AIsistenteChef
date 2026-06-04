import datetime
import json
import os
import streamlit as st
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    # Modelo IA
    model_id = 'Qwen/Qwen2.5-3B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
        load_in_4bit=True
    )
    # Modelo Embedding para RAG
    retriever = SentenceTransformer('all-MiniLM-L6-v2')
    return tokenizer, model, retriever

@st.cache_data
def cargar_y_preparar_dataset():
    # Cargamos solo la sección de entrenamiento
    ds = load_dataset('datahiveai/recipes-with-nutrition', split='train')
    
    # Convertimos a DataFrame de Pandas y limitamos a 8000 filas para optimizar
    df = pd.DataFrame(ds).head(8000)
    
    # Limpieza básica de strings para asegurar que las búsquedas no fallen por nulos
    df['recipe_name'] = df['recipe_name'].fillna('')
    df['ingredients'] = df['ingredients'].fillna('')
    df['health_labels'] = df['health_labels'].fillna('')
    df['dish_type'] = df['dish_type'].fillna('')
    df['meal_type'] = df['meal_type'].fillna('')
    df['ingredient_lines'] = df['ingredient_lines'].fillna('')
    
    return df

try:
    tokenizer, model, retriever = cargar_recursos()
    df_recetas = cargar_y_preparar_dataset()
    
    # Movemos la inicialización aquí dentro para asegurar el orden de ejecución
    @st.cache_resource
    def cargar_embeddings():
        recetas_text = [
            f"Nombre: {row['recipe_name']} | Ingredientes: {row['ingredients']} | Dieta: {row['health_labels']} | Tipo: {row['dish_type']} {row['meal_type']}"
            for _, row in df_recetas.iterrows()
        ]
        return retriever.encode(recetas_text, convert_to_numpy=True)

    embeddings_recetas = cargar_embeddings()

except Exception as e:
    st.error(f'❌ Error de inicialización: {e}')
    st.stop()


# -----------------------------------------------------------------------------
# LÓGICA RAG UPTIMIZADA
# -----------------------------------------------------------------------------
@st.cache_resource
def cargar_embeddings():
    # Creamos un texto rico combinando los campos clave solicitados
    recetas_text = [
        f"Nombre: {row['recipe_name']} | Ingredientes: {row['ingredients']} | Dieta: {row['health_labels']} | Tipo: {row['dish_type']} {row['meal_type']}"
        for _, row in df_recetas.iterrows()
    ]
    return retriever.encode(recetas_text, convert_to_numpy=True)

# Inicializar los embeddings una sola vez
embeddings_recetas = cargar_embeddings()

def obtener_contexto(query, df_filtrado, embeddings_filtrados):
    if df_filtrado.empty:
        return 'No se encontraron recetas que cumplan con los filtros de dieta o alergias.'
        
    # Vectorizar la consulta del usuario usando el retriever global
    embedding_query = retriever.encode([query], convert_to_numpy=True)
    
    # Calcular similitud solo con las filas que pasaron el filtro
    similitudes = cosine_similarity(embedding_query, embeddings_filtrados)
    
    # Tomar el top 2 o el máximo disponible si hay menos de 2
    top_k = min(2, len(df_filtrado))
    indices = np.argsort(similitudes[0])[-top_k:]
    
    contexto = '\n'.join([
        f"- {df_filtrado.iloc[i]['recipe_name']} (Tipo: {df_filtrado.iloc[i]['meal_type']}/{df_filtrado.iloc[i]['dish_type']}): {df_filtrado.iloc[i]['instructions']}"
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

# Nuevos selectores en la UI para Alergias y Dietas
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
    # --- PROCESO DE FILTRADO PRE-RAG ---
    df_filtrado = df_recetas.copy()
    indices_validos = np.arange(len(df_recetas))
    
    # 1. Filtro de Alergias (Exclusión)
    if alergias:
        lista_alergias = [a.strip().lower() for a in alergias.split(',') if a.strip()]
        for alergia in lista_alergias:
            mask = ~df_filtrado['ingredients'].str.lower().str.contains(alergia) & ~df_filtrado['health_labels'].str.lower().str.contains(alergia)
            df_filtrado = df_filtrado[mask]
            indices_validos = indices_validos[mask]

    # 2. Filtro de Tipo de Dieta (Inclusión)
    if tipo_dieta:
        for dieta in tipo_dieta:
            mask = df_filtrado['health_labels'].str.upper().str.contains(dieta)
            df_filtrado = df_filtrado[mask]
            indices_validos = indices_validos[mask]
            
    # 3. Filtro de Momento del Día
    if momento_dia != 'Cualquiera':
        mask = df_filtrado['meal_type'].str.lower().str.contains(momento_dia) | df_filtrado['dish_type'].str.lower().str.contains(momento_dia)
        df_filtrado = df_filtrado[mask]
        indices_validos = indices_validos[mask]

    # Obtener subconjunto de embeddings para el cálculo de similitud corregido
    embeddings_filtrados = embeddings_recetas[indices_validos]

    # Obtener Contexto RAG optimizado con los filtros aplicados
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
    - Formato JSON con claves: opcion_1, opcion_2, opcion_3. Cada una con nombre, tecnica, tiempo, pasos (lista).
    """

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f'Ingredientes disponibles: {ingredientes_usuario}. Momento del día deseado: {momento_dia}.'}
    ]

    with st.spinner('🧠 Cocinando con RAG y filtrado inteligente...'):
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)
        outputs = model.generate(inputs, max_new_tokens=800, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        texto_generado = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Limpieza JSON
        try:
            inicio, fin = texto_generado.find('{'), texto_generado.rfind('}') + 1
            recetas = json.loads(texto_generado[inicio:fin])
            st.session_state['recetas_generadas'] = recetas
            st.session_state['ingredientes_usados'] = ingredientes_usuario
        except:
            st.error('Error al generar o parsear el JSON de la receta.')

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
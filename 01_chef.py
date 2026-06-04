import datetime
import json
import os
import re

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# OPTIMIZACIÓN VRAM
# -----------------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
HISTORIAL_FILE = "historial_cocina.json"

# -----------------------------------------------------------------------------
# 1. CARGA DEL MODELO
# -----------------------------------------------------------------------------
@st.cache_resource
def cargar_motor_ia():

    model_id = "Qwen/Qwen2.5-3-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True
    )

    return tokenizer, model


try:
    tokenizer, model = cargar_motor_ia()

except Exception as e:
    st.error(f"❌ Error cargando modelo: {e}")
    tokenizer = None
    model = None

# -----------------------------------------------------------------------------
# 2. HISTORIAL
# -----------------------------------------------------------------------------
def cargar_historial():

    if not os.path.exists(HISTORIAL_FILE):
        return []

    try:
        with open(HISTORIAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return []


def guardar_en_historial(plato_elegido, ingredientes, tecnica):

    historial = cargar_historial()

    nuevo_registro = {
        "fecha": str(datetime.date.today()),
        "plato": plato_elegido,
        "ingredientes": ingredientes,
        "tecnica": tecnica,
    }

    historial.append(nuevo_registro)

    # Mantener solo últimos 5
    historial = historial[-5:]

    with open(HISTORIAL_FILE, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=4)

# -----------------------------------------------------------------------------
# 3. UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Chef Rapidín",
    page_icon="🍳",
    layout="centered"
)

st.title("🍳 Chef Rapidín V2")
st.subheader("IA culinaria impulsada por GPU local")

st.write(
    "Dime qué tienes en la nevera y la IA improvisará "
    "3 recetas random para sobrevivir otra noche más 💀"
)

historial_previo = cargar_historial()

# -----------------------------------------------------------------------------
# HISTORIAL VISUAL
# -----------------------------------------------------------------------------
if historial_previo:

    with st.expander("🕰️ Tus experimentos anteriores"):

        for item in reversed(historial_previo):

            st.write(
                f"• **{item['fecha']}** → "
                f"{item['plato']} | *{item['tecnica']}*"
            )

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
ingredientes_usuario = st.text_input(
    "¿Qué tienes por ahí tirado?",
    value="arroz, atún, cebolla, queso, tomate"
)

# -----------------------------------------------------------------------------
# 4. GENERACIÓN
# -----------------------------------------------------------------------------
if st.button("⚡ Generar recetas", type="primary"):

    if not ingredientes_usuario.strip():

        st.warning(
            "Introduce ingredientes o tocará cenar aire premium."
        )

    elif model is None:

        st.error("❌ Modelo no disponible.")

    else:

        texto_historial = "Ninguno"

        if historial_previo:

            texto_historial = ", ".join([
                f"{h['plato']} ({h['tecnica']})"
                for h in historial_previo
            ])

        # ---------------------------------------------------------------------
        # PROMPT
        # ---------------------------------------------------------------------
        system_prompt = f"""
Eres Chef Rapidín.

Genera 3 recetas radicalmente diferentes.

Reglas:
- Español de España
- Uno deber ser a la sarten
- Uno debe ser tipo bowl / ensalada
- Uno un invento de presentacion
- Humor sarcástico basto
- Máximo 20 minutos
- No repitas pasos
- No repitas técnicas
- No repitas verbos
- No repetir recetas del historial
- Respuesta SOLO en JSON válido
- Sin markdown
- Cada paso debe ser MUY corto
- Sin texto fuera del JSON

Historial prohibido:
{texto_historial}

Formato exacto:

{{
    "opcion_1": {{
        "nombre": "",
        "tecnica": "",
        "tiempo": "",
        "pasos": ["", "", ""]
    }},
    "opcion_2": {{
        "nombre": "",
        "tecnica": "",
        "tiempo": "",
        "pasos": ["", "", ""]
    }},
    "opcion_3": {{
        "nombre": "",
        "tecnica": "",
        "tiempo": "",
        "pasos": ["", "", ""]
    }}
}}
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Ingredientes: {ingredientes_usuario}"
            }
        ]

        # ---------------------------------------------------------------------
        # GENERACIÓN
        # ---------------------------------------------------------------------
        with st.spinner("🧠 Cocinando ideas cuestionables..."):

            try:

                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)

                outputs = model.generate(
                    inputs,
                    max_new_tokens=750,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

                texto_generado = tokenizer.decode(
                    outputs[0][inputs.shape[1]:],
                    skip_special_tokens=True
                )

                # -----------------------------------------------------------------
                # DEBUG VISUAL
                # -----------------------------------------------------------------
                with st.expander("🛠️ Debug IA"):
                    st.code(texto_generado)

                # -----------------------------------------------------------------
                # EXTRAER JSON
                # -----------------------------------------------------------------
                try:
                    inicio = texto_generado.find("{")
                    fin = texto_generado.rfind("}") + 1

                    if inicio == -1 or fin == 0:
                        raise ValueError("No se encontró JSON válido")

                    texto_json = texto_generado[inicio:fin]
                    recetas = json.loads(texto_json)
                except json.JSONDecodeError as e:
                    st.error("❌ La IA generó un JSON roto 💀")

                    st.code(texto_generado)

                    raise e

                st.session_state["recetas_generadas"] = recetas

                st.session_state[
                    "ingredientes_usados"
                ] = ingredientes_usuario

            except Exception as e:

                st.error(f"❌ Error generando recetas: {e}")

# -----------------------------------------------------------------------------
# 5. RENDER RECETAS
# -----------------------------------------------------------------------------
if "recetas_generadas" in st.session_state:

    recetas = st.session_state["recetas_generadas"]

    st.success("🍽️ Menú generado correctamente")

    col1, col2, col3 = st.columns(3)

    columnas = [col1, col2, col3]
    claves = ["opcion_1", "opcion_2", "opcion_3"]
    iconos = ["💥", "🌀", "🛸"]

    for col, clave, icono in zip(columnas, claves, iconos):

        with col:

            opc = recetas[clave]

            st.markdown(
                f"### {icono} {opc['tecnica']}"
            )

            st.write(f"**{opc['nombre']}**")

            st.caption(
                f"⏱️ Tiempo: {opc['tiempo']}"
            )

            for paso in opc["pasos"]:

                st.write(f"• {paso}")

            if st.button(
                f"Elegir {opc['tecnica']}",
                key=f"btn_{clave}"
            ):

                guardar_en_historial(
                    plato_elegido=opc["nombre"],
                    ingredientes=st.session_state.get("ingredientes_usados", "Desconocidos"),
                    tecnica=opc["tecnica"]
                )

                st.success(
                    "📚 Guardado en historial"
                )

                st.rerun()
import os
import google.generativeai as genai


def configure_gemini(api_key):
    """Configura el API de Gemini con la key proporcionada"""
    genai.configure(api_key=api_key)


def upload_image(image_path):
    """Sube una imagen a Gemini y retorna el archivo"""
    try:
        file = genai.upload_file(image_path, mime_type="image/jpeg")
        print(f"Imagen cargada: {file.display_name}")
        return file
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return None


def create_gemini_model():
    """Crea y configura el modelo de Gemini"""
    generation_config = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
    }
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )


def get_recipe_from_image(image_path, prompt, api_key):
    """
    Función principal que procesa una imagen y retorna una receta basada en el prompt
    """
    # Configurar Gemini
    configure_gemini(api_key)

    # Subir imagen
    image_file = upload_image(image_path)
    if not image_file:
        return "Error al procesar la imagen"

    # Crear modelo
    model = create_gemini_model()

    # Crear sesión de chat
    chat = model.start_chat()

    # Preparar el mensaje con la imagen y el prompt
    message = [
        image_file,
        prompt
    ]

    # Enviar mensaje y obtener respuesta
    try:
        response = chat.send_message(message)
        return response.text
    except Exception as e:
        return f"Error al procesar la solicitud: {e}"


def main():
    # Configuración
    API_KEY = "AIzaSyBeDEapDdHQN6OWRIuk5ufr-zarQru9XeU"  # Reemplaza con tu API key
    IMAGE_PATH = "huancaina-potato.jpeg"  # Reemplaza con la ruta de tu imagen

    # Prompt personalizado
    PROMPT = """
    Dado esta imagen:
    1. Primero, describe la imagen
    2. Luego, detalla la receta para preparar este plato en formato JSON. 
       Incluye los nombres de los productos y las cantidades para la receta
    """

    # Ejecutar el proceso
    result = get_recipe_from_image(IMAGE_PATH, PROMPT, API_KEY)
    print(result)


if __name__ == "__main__":
    main()
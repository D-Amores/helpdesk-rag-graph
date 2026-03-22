# main.py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rag.rag_system import VectorRAGSystem


def main():
    print("🎧 Testing VectorRAGSystem")
    print("=" * 40)

    # Crear el sistema RAG
    rag = VectorRAGSystem()

    # Probar búsqueda
    resultado = rag.search("¿cómo reseteo mi contraseña?")

    print("\n📊 Resultado:")
    print(f"Respuesta:  {resultado['response']}")
    print(f"Confianza:  {resultado['confidence']}")
    print(f"Fuentes:    {resultado['sources']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script de collecte de réponse pour l'IA Opix
Envoie la réponse mot par mot vers l'interface web
"""

import json
import time
import requests
from typing import Generator, Any


class StreamResponseCollector:
    """
    Collecte les réponses de l'IA et les envoie mot par mot
    vers une interface web via WebSocket ou HTTP streaming.
    """
    
    def __init__(self, server_url: str = "http://localhost:5000"):
        """
        Initialise le collecteur de réponse.
        
        Args:
            server_url: URL du serveur Flask (app.py)
        """
        self.server_url = server_url
        self.session = requests.Session()
        
    def send_word_by_word(self, text: str, conversation_id: str = "default") -> None:
        """
        Envoie le texte mot par mot au serveur.
        
        Args:
            text: Texte complet à envoyer
            conversation_id: Identifiant de la conversation
        """
        words = text.split()
        for word in words:
            data = {
                "conversation_id": conversation_id,
                "word": word,
                "timestamp": time.time()
            }
            try:
                response = self.session.post(
                    f"{self.server_url}/receive_word",
                    json=data,
                    timeout=1.0
                )
                if response.status_code != 200:
                    print(f"Erreur lors de l'envoi du mot '{word}': {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Erreur de connexion: {e}")
                break
            
            # Petite pause pour simuler le streaming
            time.sleep(0.05)
        
        # Envoyer un signal de fin
        end_data = {
            "conversation_id": conversation_id,
            "word": "",
            "end_of_message": True,
            "timestamp": time.time()
        }
        try:
            self.session.post(
                f"{self.server_url}/receive_word",
                json=end_data,
                timeout=1.0
            )
        except requests.exceptions.RequestException:
            pass

    def send_chunk(self, chunk: str, conversation_id: str = "default") -> None:
        """
        Envoie un fragment de texte au serveur.
        
        Args:
            chunk: Fragment de texte à envoyer
            conversation_id: Identifiant de la conversation
        """
        data = {
            "conversation_id": conversation_id,
            "chunk": chunk,
            "timestamp": time.time()
        }
        try:
            response = self.session.post(
                f"{self.server_url}/receive_chunk",
                json=data,
                timeout=1.0
            )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de l'envoi du chunk: {e}")
            return False

    def stream_from_generator(
        self, 
        generator: Generator[str, None, None], 
        conversation_id: str = "default"
    ) -> None:
        """
        Collecte les chunks depuis un générateur et les envoie au serveur.
        
        Args:
            generator: Générateur qui produit des fragments de texte
            conversation_id: Identifiant de la conversation
        """
        for chunk in generator:
            self.send_chunk(chunk, conversation_id)
        
        # Envoyer un signal de fin
        end_data = {
            "conversation_id": conversation_id,
            "chunk": "",
            "end_of_message": True,
            "timestamp": time.time()
        }
        try:
            self.session.post(
                f"{self.server_url}/receive_chunk",
                json=end_data,
                timeout=1.0
            )
        except requests.exceptions.RequestException:
            pass


# Fonction utilitaire pour tester le collecteur
if __name__ == "__main__":
    collector = StreamResponseCollector()
    
    # Test avec un texte simple
    test_text = "Bonjour, je suis l'assistant IA Opix. Comment puis-je vous aider aujourd'hui ?"
    print(f"Envoi du texte: {test_text}")
    collector.send_word_by_word(test_text, "test_conversation")
    print("Envoi terminé.")

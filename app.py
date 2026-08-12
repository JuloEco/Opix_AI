#!/usr/bin/env python3
"""
Serveur Flask pour l'interface de chat Opix AI
Gère les requêtes HTTP et le streaming des réponses
"""

import json
import time
import threading
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import torch
import sys
import os

# Ajouter le répertoire courant au path pour importer les modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_IA import (
    ModernLLM, 
    ModelArgs, 
    FrenchTokenizerWrapper,
    StreamingInferenceEngine,
    ChatFormatter
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Stockage des conversations en mémoire
conversations = {}
conversation_lock = threading.Lock()

# Variables globales pour le modèle et le moteur d'inférence
model = None
tokenizer = None
engine = None
device = "cpu"


def load_model():
    """Charge le modèle et le tokenizer"""
    global model, tokenizer, engine, device
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "chat_model.pt")
    tokenizer_path = os.path.join(script_dir, "fr_bpe_tokenizer.json")
    
    # Vérifier si les fichiers existent
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(script_dir, "best_model.pt")
    
    if not os.path.exists(tokenizer_path):
        tokenizer_path = os.path.join(script_dir, "fr_bpe_tokenizer.json")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Charger le tokenizer
        tokenizer = FrenchTokenizerWrapper(tokenizer_path)
        print(f"Tokenizer chargé avec succès (vocab size: {tokenizer.vocab_size})")
        
        # Charger le modèle
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                saved_args = checkpoint.get("args", None)
            else:
                state_dict = checkpoint
                saved_args = None
            
            if saved_args is not None and hasattr(saved_args, "vocab_size"):
                args = saved_args
                args.device = device
            else:
                emb_shape = state_dict["tok_embeddings.weight"].shape
                vocab_size_ckpt, dim_ckpt = emb_shape[0], emb_shape[1]
                
                layer_indices = set()
                for key in state_dict.keys():
                    if key.startswith("layers."):
                        idx = int(key.split(".")[1])
                        layer_indices.add(idx)
                n_layers_ckpt = len(layer_indices) if layer_indices else 8
                
                wk_shape = state_dict["layers.0.attn.wk.weight"].shape
                n_heads_ckpt = 8
                head_dim = dim_ckpt // n_heads_ckpt
                n_kv_heads_ckpt = wk_shape[0] // head_dim
                
                args = ModelArgs(
                    vocab_size=vocab_size_ckpt,
                    dim=dim_ckpt,
                    n_layers=n_layers_ckpt,
                    n_heads=n_heads_ckpt,
                    n_kv_heads=n_kv_heads_ckpt,
                    device=device,
                )
            
            model = ModernLLM(args).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            
            # Créer le moteur d'inférence
            engine = StreamingInferenceEngine(model, tokenizer)
            
            print(f"Modèle chargé avec succès sur {device}")
            return True
        else:
            print(f"Fichier modèle introuvable: {checkpoint_path}")
            return False
            
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Vérifie l'état du serveur"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': device
    })


@app.route('/send_message', methods=['POST'])
def send_message():
    """
    Reçoit un message de l'utilisateur et lance la génération de réponse
    """
    data = request.get_json()
    conversation_id = data.get('conversation_id', 'default')
    message = data.get('message', '')
    
    if not message:
        return jsonify({'error': 'Message vide'}), 400
    
    # Initialiser la conversation si elle n'existe pas
    with conversation_lock:
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        # Ajouter le message utilisateur
        conversations[conversation_id].append({
            'role': 'user',
            'content': message
        })
    
    # Lancer la génération en arrière-plan
    def generate_response():
        history = conversations[conversation_id]
        
        # Générer la réponse
        response_text = ""
        for chunk in engine.stream_generate_from_history(
            history=history,
            max_new_tokens=80,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.3,
        ):
            response_text += chunk
            
            # Envoyer le chunk via Server-Sent Events
            # (géré par l'endpoint /stream)
            with conversation_lock:
                if conversation_id in conversations:
                    conversations[conversation_id].append({
                        'role': 'assistant',
                        'content': response_text,
                        'partial': True
                    })
        
        # Ajouter la réponse complète
        with conversation_lock:
            if conversation_id in conversations:
                # Retirer les messages partiels
                conversations[conversation_id] = [
                    msg for msg in conversations[conversation_id] 
                    if not msg.get('partial', False)
                ]
                conversations[conversation_id].append({
                    'role': 'assistant',
                    'content': response_text
                })
    
    # Lancer la génération dans un thread séparé
    thread = threading.Thread(target=generate_response)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'processing',
        'conversation_id': conversation_id
    })


@app.route('/receive_word', methods=['POST'])
def receive_word():
    """
    Reçoit un mot depuis le collecteur de réponse (stream_response.py)
    """
    data = request.get_json()
    conversation_id = data.get('conversation_id', 'default')
    word = data.get('word', '')
    end_of_message = data.get('end_of_message', False)
    
    # Stocker le mot dans la conversation
    with conversation_lock:
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        if end_of_message:
            # Marquer la fin du message
            if conversations[conversation_id]:
                conversations[conversation_id][-1]['end_of_message'] = True
        else:
            # Ajouter le mot au dernier message
            if conversations[conversation_id]:
                last_msg = conversations[conversation_id][-1]
                if 'content' in last_msg:
                    last_msg['content'] += ' ' + word
                else:
                    last_msg['content'] = word
            else:
                conversations[conversation_id].append({
                    'role': 'assistant',
                    'content': word
                })
    
    return jsonify({'status': 'ok'})


@app.route('/receive_chunk', methods=['POST'])
def receive_chunk():
    """
    Reçoit un fragment de texte depuis le collecteur de réponse
    """
    data = request.get_json()
    conversation_id = data.get('conversation_id', 'default')
    chunk = data.get('chunk', '')
    end_of_message = data.get('end_of_message', False)
    
    # Stocker le chunk dans la conversation
    with conversation_lock:
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        if end_of_message:
            # Marquer la fin du message
            if conversations[conversation_id]:
                conversations[conversation_id][-1]['end_of_message'] = True
        else:
            # Ajouter le chunk au dernier message
            if conversations[conversation_id]:
                last_msg = conversations[conversation_id][-1]
                if 'content' in last_msg:
                    last_msg['content'] += chunk
                else:
                    last_msg['content'] = chunk
            else:
                conversations[conversation_id].append({
                    'role': 'assistant',
                    'content': chunk
                })
    
    return jsonify({'status': 'ok'})


@app.route('/stream')
def stream():
    """
    Endpoint pour Server-Sent Events (SSE)
    Envoie les mises à jour de la conversation en temps réel
    """
    conversation_id = request.args.get('conversation_id', 'default')
    
    def event_stream():
        last_index = 0
        
        while True:
            with conversation_lock:
                if conversation_id in conversations:
                    conv = conversations[conversation_id]
                    
                    # Envoyer les nouveaux messages
                    for i in range(last_index, len(conv)):
                        msg = conv[i]
                        
                        if msg['role'] == 'assistant':
                            # Envoyer le message complet ou partiel
                            yield f"data: {json.dumps({'type': 'chunk', 'chunk': msg.get('content', '')})}\n\n"
                            
                        elif msg['role'] == 'user':
                            # Envoyer le message utilisateur
                            yield f"data: {json.dumps({'type': 'start', 'text': msg.get('content', '')})}\n\n"
                        
                        last_index = i + 1
                    
                    # Vérifier si le dernier message est terminé
                    if conv and conv[-1].get('end_of_message'):
                        yield f"data: {json.dumps({'type': 'end'})}\n\n"
                        last_index = len(conv)
            
            # Attendre un peu avant la prochaine vérification
            time.sleep(0.1)
    
    return Response(
        event_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Cache-Control'
        }
    )


@app.route('/get_conversation', methods=['GET'])
def get_conversation():
    """Récupère l'historique d'une conversation"""
    conversation_id = request.args.get('conversation_id', 'default')
    
    with conversation_lock:
        conv = conversations.get(conversation_id, [])
    
    # Filtrer les messages partiels
    clean_conv = [
        {'role': msg['role'], 'content': msg.get('content', '')}
        for msg in conv if not msg.get('partial', False)
    ]
    
    return jsonify({
        'conversation_id': conversation_id,
        'messages': clean_conv
    })


@app.route('/new_conversation', methods=['POST'])
def new_conversation():
    """Crée une nouvelle conversation"""
    conversation_id = f"conv_{int(time.time())}_{len(conversations)}"
    
    with conversation_lock:
        conversations[conversation_id] = []
    
    return jsonify({
        'conversation_id': conversation_id,
        'status': 'created'
    })


@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Efface une conversation"""
    data = request.get_json()
    conversation_id = data.get('conversation_id', 'default')
    
    with conversation_lock:
        if conversation_id in conversations:
            del conversations[conversation_id]
    
    return jsonify({'status': 'cleared'})


if __name__ == '__main__':
    # Charger le modèle au démarrage
    print("Chargement du modèle...")
    if load_model():
        print("Modèle chargé avec succès !")
    else:
        print("Avertissement: Impossible de charger le modèle. Le serveur démarrera quand même.")
    
    # Démarrer le serveur
    print("\nDémarrage du serveur Flask...")
    print("Interface disponible à: http://localhost:5000")
    print("Ouvrez chat_interface.html dans votre navigateur")
    
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)

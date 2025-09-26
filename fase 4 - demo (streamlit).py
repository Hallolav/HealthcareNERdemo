import streamlit as st
import pickle
import torch
import torch.nn as nn
import numpy as np

# CRF Layer Implementation
class CRF(nn.Module):
    def __init__(self, num_tags, batch_first=True):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # Transition parameters: transitions[i][j] is the score of transitioning from j to i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Start and end transitions
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
    
    def forward(self, emissions, tags, mask=None):
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        log_likelihood = self._compute_log_likelihood(emissions, tags, mask)
        # Return the mean negative log likelihood as a scalar loss
        return -log_likelihood.mean()
    
    def decode(self, emissions, mask=None):
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        return self._viterbi_decode(emissions, mask)
    
    def _compute_log_likelihood(self, emissions, tags, mask):
        seq_length, batch_size = emissions.shape[:2]
        
        if mask is None:
            mask = torch.ones((seq_length, batch_size), dtype=torch.bool, device=emissions.device)
        
        # Compute the log sum of all possible paths
        log_denominator = self._compute_log_partition_function(emissions, mask)
        
        # Compute the log probability of the given path
        log_numerator = self._compute_score(emissions, tags, mask)
        
        return log_numerator - log_denominator
    
    def _compute_score(self, emissions, tags, mask):
        seq_length, batch_size = emissions.shape[:2]
        score = torch.zeros(batch_size, device=emissions.device)
        
        # Add start transition scores
        score += self.start_transitions[tags[0]]
        
        # Add emission scores
        for i in range(seq_length):
            score += emissions[i].gather(1, tags[i].unsqueeze(1)).squeeze(1) * mask[i]
        
        # Add transition scores
        for i in range(1, seq_length):
            transition_score = self.transitions[tags[i], tags[i-1]]
            score += transition_score * mask[i]
        
        # Add end transition scores
        last_tag_indices = mask.sum(0) - 1
        last_tags = tags.gather(0, last_tag_indices.unsqueeze(0)).squeeze(0)
        score += self.end_transitions[last_tags]
        
        return score
    
    def _compute_log_partition_function(self, emissions, mask):
        seq_length, batch_size, num_tags = emissions.shape
        
        # Initialize with start transitions
        log_alpha = self.start_transitions.view(1, -1) + emissions[0]
        
        for i in range(1, seq_length):
            # Broadcast log_alpha and transitions
            broadcast_log_alpha = log_alpha.unsqueeze(2)
            broadcast_transitions = self.transitions.unsqueeze(0)
            broadcast_emissions = emissions[i].unsqueeze(1)
            
            # Compute scores
            inner = broadcast_log_alpha + broadcast_transitions + broadcast_emissions
            
            # Log-sum-exp over the previous states
            log_alpha_new = torch.logsumexp(inner, dim=1)
            
            # Update only valid positions
            log_alpha = torch.where(mask[i].unsqueeze(1), log_alpha_new, log_alpha)
        
        # Add end transitions
        log_alpha += self.end_transitions.view(1, -1)
        
        return torch.logsumexp(log_alpha, dim=1)
    
    def _viterbi_decode(self, emissions, mask):
        seq_length, batch_size, num_tags = emissions.shape
        
        if mask is None:
            mask = torch.ones((seq_length, batch_size), dtype=torch.bool, device=emissions.device)
        
        # Initialize
        log_prob = self.start_transitions.view(1, -1) + emissions[0]
        path_indices = []
        
        # Forward pass
        for i in range(1, seq_length):
            broadcast_log_prob = log_prob.unsqueeze(2)
            broadcast_transitions = self.transitions.unsqueeze(0)
            broadcast_emissions = emissions[i].unsqueeze(1)
            
            # Compute scores
            scores = broadcast_log_prob + broadcast_transitions + broadcast_emissions
            
            # Find best previous states
            log_prob_new, path_idx = torch.max(scores, dim=1)
            path_indices.append(path_idx)
            
            # Update only valid positions
            log_prob = torch.where(mask[i].unsqueeze(1), log_prob_new, log_prob)
        
        # Add end transitions
        log_prob += self.end_transitions.view(1, -1)
        
        # Find best final states
        _, best_last_tags = torch.max(log_prob, dim=1)
        
        # Backward pass to get the best path
        best_paths = []
        for batch_idx in range(batch_size):
            best_path = [best_last_tags[batch_idx].item()]
            
            for path_idx in reversed(path_indices):
                best_path.append(path_idx[batch_idx, best_path[-1]].item())
            
            best_path.reverse()
            best_paths.append(best_path)
        
        return best_paths

# BiLSTM-CRF Model
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, num_layers=2, dropout=0.1, word_to_id=None):
        super(BiLSTM_CRF, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.num_layers = num_layers
        
        # Embedding layer
        padding_idx = word_to_id.get('<PAD>', 0) if word_to_id else 0
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer to map LSTM output to tag space
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        
        # CRF layer
        self.crf = CRF(num_tags, batch_first=True)
    
    def forward(self, sentences, tags=None, mask=None):
        # Embedding
        embeds = self.embedding(sentences)
        
        # LSTM
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        
        # Linear layer
        emissions = self.hidden2tag(lstm_out)
        
        if tags is not None:
            # Training mode: return loss
            return self.crf(emissions, tags, mask)
        else:
            # Inference mode: return best path
            return self.crf.decode(emissions, mask)

@st.cache_resource
def load_model():
    """Load the trained BiLSTM-CRF model and mappings"""
    try:
        # Load checkpoint
        checkpoint = torch.load('bilstm_crf_final_model.pt', map_location=torch.device('cpu'))
        
        # Extract word_to_id first for padding index
        word_to_id = checkpoint['word_to_id']
        
        # Extract model config
        config = checkpoint['model_config']
        
        # Initialize model with padding index from word_to_id
        model = BiLSTM_CRF(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_tags=config['num_tags'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            word_to_id=word_to_id
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Restore other mappings and metrics
        id_to_word = checkpoint['id_to_word']
        label_to_id = checkpoint['label_to_id']
        id_to_label = checkpoint['id_to_label']
        entity_f1 = checkpoint['entity_f1']
        
        return model, word_to_id, id_to_label, entity_f1
    
    except FileNotFoundError:
        st.error("Model file 'bilstm_crf_final_model.pt' not found. Please make sure the model file is in the same directory.")
        return None, None, None, None

def predict_sentence(model, sentence, word_to_id, id_to_label, device='cpu', max_length=128):
    """Predict entities in a sentence using the trained model"""
    model.eval()
    
    tokens = sentence.split()
    token_ids = [word_to_id.get(token, word_to_id['<UNK>']) for token in tokens]
    
    # Pad or truncate
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
        tokens = tokens[:max_length]
    
    attention_mask = [1] * len(token_ids)
    
    # Pad
    padding_length = max_length - len(token_ids)
    token_ids += [word_to_id['<PAD>']] * padding_length
    attention_mask += [0] * padding_length
    
    # Convert to tensors
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    mask = torch.tensor([attention_mask], dtype=torch.bool).to(device)
    
    with torch.no_grad():
        predictions = model(input_ids, mask=mask)
    
    # Get predictions for actual tokens only
    pred_labels = [id_to_label[p] for p in predictions[0][:len(tokens)]]
    
    return list(zip(tokens, pred_labels))

def main():
    st.title("ilionx Healthcare Demo symptoon-NER Demo")
    st.write("""Voor je ligt een demo van de Named Entity Recognition (NER) tool van ilionx. Deze tool herkent symptomen en personen in tekst. Op deze manier is het voor artsen snel duidelijk welke symptomen genoemd worden in een intakegesprek. Op deze manier wordt het (geautomatiseerd) schrijven én doorlezen van een EPD een stuk makkelijker.
 
Het NER-model is getraind op fictieve data. Daardoor kan de tool gedeeld worden met derden zonder dat er privacygevoelige informatie wordt gedeeld. Het nadeel hiervan is dat de data onnatuurlijk lijkt: mensen praten niet zoals de door de teksten die door een gegenereerd zijn. De "echte" EPD's zullen van hogere kwaliteit zijn dan de gegenereerde teksten. Daardoor zullen de daadwerkelijke resultaten beter zijn dan deze demo-resultaten.
 
Het voordeel van een NER-model ten opzichte van een LLM is dat het model veel kleiner is. Het draait lokaal en gevoelige (patiënt)gegevens worden dus niet met derden gedeeld. Ook werkt het NER-model veel sneller: het kan in real-time teksten analyseren en de symptomen weergeven.
 
Om je eigen tekst te kunnen testen, kun je de voorbeeldtekst hieronder aanpassen. Houd er rekening mee dat het model een bepaalde opmaak verwacht: een cliëntnummer, naam, arts en datum bovenaan de pagina. Daarnaast wordt het resultaat als een "gesprek" weergegeven, dus de patient zegt iets en dan de dokter. Zo gaat het immers ook "in het echt", wanneer tijdens een intakegesprek de patient en de huisarts met elkaar praten.""")
    
    # Load model
    model, word_to_id, id_to_label, entity_f1 = load_model()
    
    if model is None:
        st.stop()

    
    # Default example text
    default_text = """clientnummer: 1000\nnaam: Aisha Bakari\ndokter: Oliveira\ndatum: 15-09-2023\n\ndokter: Goedemiddag, Aisha. Wat brengt u vandaag?\npatient: Hallo, ik voel me de laatste tijd erg futloos en ziek.\ndokter: Hoe lang ervaart u deze klachten al?\npatient: Het begon ongeveer twee weken geleden.\ndokter: Heeft u daarnaast nog andere symptomen, zoals koorts of pijn?\npatient: Nee, geen koorts, maar ik voel me heel zwak en erg moe de hele dag door.\n\ndokter: Zijn er dingen die u niet meer kunt doen door deze klachten?\npatient: Ja, ik heb weinig energie en kan bijna niets meer ondernemen.\ndokter: Heeft iemand in uw familie soortgelijke klachten gehad of problemen met vermoeidheid?\npatient: Mijn moeder zegt dat zij zich vroeger ook wel eens erg moe voelde, maar niet zo extreem.
dokter: Voelt u zich ook ziek, zoals dat u zich niet goed voelt?
patient: Ja, ik voel me erg ziek, alsof ik een griep heb, maar ik heb geen koorts.
dokter: Heeft u last van hoofdpijn of spierpijn?
patient: Een beetje spierpijn, vooral in mijn benen, maar geen hoofdpijn.
dokter: Hoe is uw eetlust?
patient: Mijn eetlust is verminderd, ik heb geen zin om te eten.
dokter: Heeft u recent reizen gemaakt of contact gehad met ziektes?
patient: Nee, ik ben niet recent ergens geweest en heb geen contact met zieke mensen.
dokter: Is er iets in uw familie dat u zorgen baart?
patient: Niet dat ik weet, mijn familie is gezond.
dokter: Dank u, Aisha. We gaan verder onderzoeken om de oorzaak te achterhalen."""
    
    # Text input
    text_input = st.text_area(
        "EPD-verslag om te analyseren:",
        value=default_text,
        height=700
    )
    
    if st.button("Analyseer", type="primary"):
        if text_input.strip():
            with st.spinner("Aan het analyseren..."):
                # Predict entities
                predictions = predict_sentence(model, text_input, word_to_id, id_to_label, 'cpu')
                
                # Extract and combine entities (B- and I- tags)
                entities = []
                current_entity = []
                current_label = None
                
                for token, label in predictions:
                    if label.startswith('B-'):
                        # Start of new entity
                        if current_entity:
                            # Save previous entity
                            entity_text = ' '.join(current_entity)
                            entities.append(f"{entity_text} ({current_label})")
                        # Start new entity
                        current_entity = [token]
                        current_label = label[2:]  # Remove 'B-' prefix
                    elif label.startswith('I-') and current_entity and label[2:] == current_label:
                        # Continue current entity
                        current_entity.append(token)
                    else:
                        # End current entity if exists
                        if current_entity:
                            entity_text = ' '.join(current_entity)
                            entities.append(f"{entity_text} ({current_label})")
                            current_entity = []
                            current_label = None
                        
                        # If it's a single token entity (not O)
                        if label != 'O' and not label.startswith('I-'):
                            entities.append(f"{token} ({label})")
                
                # Don't forget the last entity
                if current_entity:
                    entity_text = ' '.join(current_entity)
                    entities.append(f"{entity_text} ({current_label})")
                
                # Display results
                st.subheader("Resultaten")
                
                # Show highlighted text (simple version)
                st.subheader("Geannoteerde tekst")
                highlighted_text = ""
                for token, label in predictions:
                    if label != 'O':
                        highlighted_text += f"**{token}** "
                    else:
                        highlighted_text += f"{token} "
                
                st.markdown(highlighted_text)

                # Show entities found
                if entities:
                    st.write("**De volgende entiteiten zijn door het NER-model gevonden:**")
                    for entity in entities:
                        st.write(f"- {entity}")
                else:
                    st.write("**Geen entiteiten gevonden**")
        else:
            st.warning("Voer alstublieft wat tekst in om te analyseren.")

if __name__ == "__main__":
    main()

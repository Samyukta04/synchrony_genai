import os
import json
import fitz
import spacy
import numpy as np
from datetime import datetime
import ollama
import re
from typing import List, Dict, Any

class PDFProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self._embedding_model_available = None
        self._tried_embedding_models = False
    
    def check_available_models(self):
        """Check which embedding models are available in Ollama"""
        try:
            models = ollama.list()
            available_models = [model['name'] for model in models['models']]
            print(f"Available Ollama models: {available_models}")
            return available_models
        except Exception as e:
            print(f"Error checking available models: {e}")
            return []

    def fallback_embedding(self, text: str, dim=384) -> np.ndarray:
        """Create a more sophisticated text-based embedding when Ollama embeddings fail"""
        import re
        from collections import Counter
        
        text_clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text_clean.split()
        
        features = np.zeros(dim)
        
        if not words:
            return features.astype(np.float32)

        word_freq = Counter(words)
        total_words = len(words)
        
        # Enhanced keyword sets
        travel_keywords = {
            'city', 'cities', 'destination', 'location', 'place', 'visit', 'explore',
            'restaurant', 'hotel', 'accommodation', 'stay', 'dining', 'eat', 'food',
            'activity', 'activities', 'adventure', 'beach', 'attraction', 'tour',
            'culture', 'history', 'tradition', 'museum', 'art', 'heritage',
            'nightlife', 'bar', 'club', 'entertainment', 'music', 'dance',
            'travel', 'trip', 'vacation', 'holiday', 'journey', 'planning',
            'tips', 'advice', 'guide', 'recommendation', 'suggestion',
            'group', 'friends', 'college', 'student', 'young', 'budget',
            'day', 'days', 'itinerary', 'schedule', 'time', 'duration'
        }
        
        cuisine_keywords = {
            'cuisine', 'cooking', 'wine', 'taste', 'flavor', 'recipe', 'chef',
            'market', 'local', 'traditional', 'specialty', 'dish', 'meal'
        }
        
        keyword_score = 0
        cuisine_score = 0
        
        for word, freq in word_freq.items():
            word_hash = hash(word) % (dim - 20)  
            features[word_hash] += freq / total_words
            
            if word in travel_keywords:
                keyword_score += freq
            if word in cuisine_keywords:
                cuisine_score += freq
        
        # Statistical features
        features[-20] = len(words) / 100.0  
        features[-19] = len(set(words)) / len(words)  
        features[-18] = keyword_score / total_words  
        features[-17] = cuisine_score / total_words  
        
        # Bigrams
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        for bigram in bigrams[:20]:  
            bigram_hash = hash(bigram) % (dim - 20)
            features[bigram_hash] += 1.0 / len(bigrams)
        
        # Structural features
        features[-16] = text.count(':') / len(text) 
        features[-15] = text.count('.') / len(text) 
        features[-14] = text.count('!') / len(text)  
        features[-13] = len([w for w in words if w.isupper()]) / total_words  
        
        # Category indicators
        if any(keyword in text.lower() for keyword in ['restaurant', 'hotel', 'accommodation']):
            features[-12] = 1.0
        if any(keyword in text.lower() for keyword in ['beach', 'activity', 'adventure']):
            features[-11] = 1.0
        if any(keyword in text.lower() for keyword in ['nightlife', 'bar', 'entertainment']):
            features[-10] = 1.0
        if any(keyword in text.lower() for keyword in ['history', 'culture', 'tradition']):
            features[-9] = 1.0
        if any(keyword in text.lower() for keyword in ['tip', 'advice', 'packing']):
            features[-8] = 1.0
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings for text using Ollama or fallback method"""
        if not self._tried_embedding_models:
            embedding_models = [
                'nomic-embed-text',
                'all-minilm', 
                'mxbai-embed-large',
                'snowflake-arctic-embed',
                'bge-large'
            ]
            
            for model in embedding_models:
                try:
                    res = ollama.embeddings(model=model, prompt="test")
                    self._embedding_model_available = model
                    print(f"Using embedding model: {model}")
                    break
                except Exception:
                    continue
            
            self._tried_embedding_models = True
            if self._embedding_model_available is None:
                print("No embedding models available. Using fallback similarity method.")

        if self._embedding_model_available:
            try:
                res = ollama.embeddings(model=self._embedding_model_available, prompt=text)
                return np.array(res["embedding"], dtype=np.float32)
            except Exception:
                self._embedding_model_available = None
        
        return self.fallback_embedding(text)

    def embed_texts(self, texts):
        """Generate embeddings for multiple texts"""
        embeddings = []
        print(f"Generating embeddings for {len(texts)} sections...")
        
        for i, t in enumerate(texts):
            if i % 20 == 0:  
                print(f"Progress: {i}/{len(texts)}")
            try:
                embeddings.append(self.embed_text(t))
            except Exception as e:
                print(f"Error embedding text {i}: {e}")
                embeddings.append(self.fallback_embedding(t))
        
        print(f"Completed: {len(embeddings)}/{len(texts)}")
        return np.vstack(embeddings)

    def is_heading(self, span, text):
        """Detect if a text span is likely a heading"""
        if not text or len(text.strip()) < 3:
            return False
        
        font_size = span.get("size", 0)
        text_clean = text.strip()
        
        if (font_size > 14 or  
            span.get("flags", 0) & 2**4 or  
            text_clean.endswith(":") and len(text_clean.split()) <= 8 or  
            text_clean.isupper() and len(text_clean.split()) <= 6 or  
            re.match(r'^[A-Z][a-z]+ [A-Za-z\s]+$', text_clean) and len(text_clean.split()) <= 8):  
            return True
        
        return False

    def extract_sections(self, pdf_path):
        """Extract sections from a PDF file"""
        doc = fitz.open(pdf_path)
        sections = []
        
        for idx, page in enumerate(doc):
            page_num = idx + 1
            blocks = page.get_text("dict")["blocks"]
            
            current_section = {"heading": None, "content": "", "page_num": page_num}
            page_headings = []
            page_content = []
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    line_text = ""
                    line_spans = []
                    
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        if text:
                            line_text += text + " "
                            line_spans.append((span, text))
                    
                    line_text = line_text.strip()
                    if not line_text:
                        continue
                    
                    is_line_heading = False
                    if line_spans:
                        first_span, first_text = line_spans[0]
                        if self.is_heading(first_span, line_text):
                            is_line_heading = True
                            page_headings.append(line_text)
                    
                    if not is_line_heading:
                        page_content.append(line_text)
            
            full_content = " ".join(page_content)
            
            if full_content.strip():
                if page_headings:
                    title = max(page_headings, key=len)
                else:
                    # Smart title generation based on content
                    content_words = full_content.lower().split()
                    if any(word in content_words for word in ['restaurant', 'hotel', 'accommodation']):
                        title = "Restaurants and Hotels"
                    elif any(word in content_words for word in ['city', 'cities', 'town']):
                        title = "Cities and Destinations"
                    elif any(word in content_words for word in ['food', 'cuisine', 'cooking', 'wine']):
                        title = "Culinary Experiences"
                    elif any(word in content_words for word in ['activity', 'activities', 'adventure', 'beach']):
                        title = "Activities and Adventures"
                    elif any(word in content_words for word in ['nightlife', 'bar', 'club', 'entertainment']):
                        title = "Nightlife and Entertainment"
                    elif any(word in content_words for word in ['packing', 'tips', 'travel']):
                        title = "Travel Tips and Packing"
                    elif any(word in content_words for word in ['history', 'culture', 'tradition']):
                        title = "History and Culture"
                    else:
                        title = f"Page {page_num} Content"
                
                sections.append({
                    "document": os.path.basename(pdf_path),
                    "page_number": page_num,
                    "section_title": title,
                    "content": full_content.strip()
                })
        
        doc.close()
        return sections

    def rank_sections(self, sections, query, top_n=5):
        """Rank sections based on similarity to query"""
        if not sections:
            return []
        
        section_texts = []
        for s in sections:
            enhanced_text = f"{s['section_title']} {s['content']}"
            section_texts.append(enhanced_text)
        
        section_embeddings = self.embed_texts(section_texts)
        query_embedding = self.embed_text(query)
        
        # Normalize embeddings
        section_embeddings = section_embeddings / np.linalg.norm(section_embeddings, axis=1, keepdims=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        similarities = np.dot(section_embeddings, query_embedding)
        
        # Apply diversity scoring
        document_counts = {}
        for i, section in enumerate(sections):
            doc_name = section["document"]
            document_counts[doc_name] = document_counts.get(doc_name, 0) + 1

        final_scores = []
        doc_appearance_count = {}
        
        for i, sim_score in enumerate(similarities):
            doc_name = sections[i]["document"]
            doc_appearance_count[doc_name] = doc_appearance_count.get(doc_name, 0) + 1
            
            diversity_penalty = 0.1 * doc_appearance_count[doc_name]
            final_score = sim_score - diversity_penalty
            
            sections[i]["score"] = float(final_score)
            final_scores.append((i, final_score))
        
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sections with diversity
        result = []
        used_docs = []
        
        for idx, score in final_scores:
            if len(result) >= top_n:
                break
            
            doc_name = sections[idx]["document"]
            
            if used_docs.count(doc_name) < 2:
                result.append(sections[idx])
                used_docs.append(doc_name)
        
        # Fill remaining slots if needed
        if len(result) < top_n:
            for idx, score in final_scores:
                if len(result) >= top_n:
                    break
                if sections[idx] not in result:
                    result.append(sections[idx])
        
        return result[:top_n]

    def summarize_text(self, text):
        """Create a better summary by taking meaningful sentences"""
        if not text:
            return ""
        
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        
        summary_sentences = []
        char_count = 0
        
        for sentence in sentences[:5]:
            if char_count + len(sentence) > 500:  
                break
            summary_sentences.append(sentence)
            char_count += len(sentence)
        
        if not summary_sentences and text:
            return text[:300].strip() + ("..." if len(text) > 300 else "")
        
        return " ".join(summary_sentences).strip()

    def process_files(self, pdf_files: List[str], query: str) -> Dict[str, Any]:
        """Main processing function for the web app"""
        print(f"Processing {len(pdf_files)} PDF files...")
        
        all_sections = []
        processed_files = []
        
        for pdf_path in pdf_files:
            print(f"Processing {os.path.basename(pdf_path)}...")
            try:
                sections = self.extract_sections(pdf_path)
                all_sections.extend(sections)
                processed_files.append(os.path.basename(pdf_path))
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                continue

        print(f"Extracted {len(all_sections)} sections total")
        
        if not all_sections:
            return {
                "error": "No content could be extracted from the uploaded PDFs",
                "metadata": {
                    "processing_timestamp": datetime.now().isoformat(),
                    "query": query,
                    "files_processed": processed_files
                }
            }

        # Rank sections
        top_sections = self.rank_sections(all_sections, query, top_n=5)
        
        # Generate answer from top sections
        answer_parts = []
        for i, section in enumerate(top_sections):
            summary = self.summarize_text(section["content"])
            if summary:
                answer_parts.append(f"From {section['document']} (Page {section['page_number']}):\n{summary}")
        
        final_answer = "\n\n".join(answer_parts)
        
        result = {
            "query": query,
            "answer": final_answer,
            "metadata": {
                "input_documents": processed_files,
                "sections_analyzed": len(all_sections),
                "top_sections_used": len(top_sections),
                "processing_timestamp": datetime.now().isoformat()
            },
            "detailed_sections": [
                {
                    "document": sec["document"],
                    "section_title": sec["section_title"],
                    "page_number": sec["page_number"],
                    "relevance_score": sec.get("score", 0),
                    "summary": self.summarize_text(sec["content"]),
                    "full_content": sec["content"]
                }
                for sec in top_sections
            ]
        }
        
        return result
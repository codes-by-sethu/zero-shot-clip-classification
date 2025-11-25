import torch
import open_clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class CLIPClassifier:
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k'):
        """
        Initialize CLIP model
        
        Args:
            model_name: CLIP model architecture
            pretrained: Pretrained weights
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and preprocessing
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=self.device
        )
        
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print(f"Loaded CLIP model: {model_name} with {pretrained} weights")
    
    def encode_text(self, text_prompts):
        """
        Encode text prompts into embeddings
        
        Args:
            text_prompts: List of text prompts
            
        Returns:
            Text embeddings
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        # Tokenize and encode text
        text_tokens = self.tokenizer(text_prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def encode_image(self, image):
        """
        Encode image into embedding
        
        Args:
            image: PIL Image or image path
            
        Returns:
            Image embedding
        """
        if isinstance(image, str):
            image = Image.open(image)
        
        # Preprocess and encode image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def classify(self, image, text_prompts):
        """
        Classify image using text prompts
        
        Args:
            image: PIL Image or image path
            text_prompts: List of text prompts
            
        Returns:
            Dictionary with classification results
        """
        # Encode image and text
        image_features = self.encode_image(image)
        text_features = self.encode_text(text_prompts)
        
        # Calculate similarity
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        similarity_scores = similarity.cpu().numpy()[0]
        
        # Get best match
        best_idx = np.argmax(similarity_scores)
        best_score = similarity_scores[best_idx]
        best_prompt = text_prompts[best_idx]
        
        return {
            'scores': similarity_scores,
            'best_class': best_prompt,
            'best_score': best_score,
            'all_prompts': text_prompts
        }
    
    def visualize_similarity(self, similarity_scores, text_prompts, save_path=None):
        """
        Visualize similarity scores
        
        Args:
            similarity_scores: Array of similarity scores
            text_prompts: List of text prompts
            save_path: Path to save visualization
        """
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(text_prompts))
        
        plt.barh(y_pos, similarity_scores)
        plt.yticks(y_pos, text_prompts)
        plt.xlabel('Similarity Score')
        plt.title('CLIP Text-Image Similarity')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Similarity visualization saved to: {save_path}")
        
        plt.show()

# Test function
def test_clip_classifier():
    """Test the CLIP classifier with a sample image"""
    classifier = CLIPClassifier()
    
    # Test with sample prompts
    text_prompts = [
        "a photo of a cat",
        "a photo of a dog", 
        "a photo of a car",
        "a photo of a person"
    ]
    
    # You can use any test image URL or local path
    test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    
    print("Testing CLIP classifier...")
    results = classifier.classify(test_image_url, text_prompts)
    
    print(f"Best match: {results['best_class']} (score: {results['best_score']:.3f})")
    print("All scores:")
    for prompt, score in zip(results['all_prompts'], results['scores']):
        print(f"  {prompt}: {score:.3f}")
    
    # Visualize results
    classifier.visualize_similarity(results['scores'], text_prompts)

if __name__ == "__main__":
    test_clip_classifier()
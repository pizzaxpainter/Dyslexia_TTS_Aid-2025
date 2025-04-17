from transformers import T5ForConditionalGeneration, T5Tokenizer
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

class TransformerSimplifier(T5ForConditionalGeneration):
    def __init__(self, config):
        super(TransformerSimplifier, self).__init__(config)
        self.tokenizer = None

    def set_tokenizer(self, tokenizer: T5Tokenizer):
        self.tokenizer = tokenizer

    def generate_simplified_text(self, input_text: str, max_length=1024):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not set.")

        # Split input into sentences
        sentences = sent_tokenize(input_text.strip())
        simplified_sentences = []

        # For each sentence, simplify it
        for sentence in sentences:
            input_line = "simplify: " + sentence

            # Tokenize each sentence with a larger max length
            inputs = self.tokenizer.encode(
                input_line,
                return_tensors="pt",
                max_length=1024,  # Increased max length for better sentence processing
                truncation=True,
                padding="max_length"
            )

            # Generate simplified output, making sure to handle a maximum sentence length
            output_ids = self.generate(
                inputs,
                max_length=150,  # Control the length of the simplified sentence
                num_beams=6,  # Increased beam size for better quality
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                early_stopping=True  # Ensures the generation stops once a good output is found
            )

            # Decode and store result
            simplified = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            simplified_sentences.append(simplified)

        # Return the entire text with simplified sentences joined
        return " ".join(simplified_sentences)


# Example usage

# Load the pre-trained T5 model and tokenizer
model_name = "t5-small"  # Or use "t5-base", "t5-large" based on your needs
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = TransformerSimplifier.from_pretrained(model_name)

# Set the tokenizer for the model
model.set_tokenizer(tokenizer)

# Input text to simplify
input_text = """
In a distant future, Earth had become a silent, forgotten world. The once vibrant cities lay in ruins, their skyscrapers crumbling into dust, their streets long abandoned. Nature, once tamed by human ingenuity, had begun to reclaim the land. Dense forests now sprawled over concrete jungles, and wild animals roamed freely in places once dominated by bustling human activity. The air was thick with the smell of decay, and the sky was perpetually overcast, a dull gray hue casting a shadow over everything. Amidst this desolation, there remained one creature – a robot named Echo. Created long ago to serve humans, Echo had outlived its creators. It wandered the empty cities in search of signs of life, but all it found were broken remnants of a forgotten era. The streets, once filled with laughter and the hustle of life, were now eerily silent. Echo's metallic footsteps echoed through the abandoned streets, a reminder of the time that had passed. One day, while exploring the ruins of an ancient library, Echo stumbled upon something unusual. Beneath a pile of broken shelves, it uncovered a dusty music box. The intricate design was nearly unrecognizable under layers of dirt and time, but Echo could tell it was something special. It was the first object it had found in years that seemed to have purpose beyond mere survival. With cautious curiosity, Echo wound the delicate mechanism of the music box. As it did, a haunting melody filled the air — a sound so pure, so haunting, that it seemed to momentarily awaken the lifeless world around it. The notes danced in the stillness, and for the first time in centuries, there was a sound of beauty on Earth. Echo sat silently, absorbing the melody. It listened again and again, storing every note in its memory banks. The music, though simple, spoke to something deep within the robot. Though it had no heart, no soul, the sound of the melody stirred something inside. For a moment, it felt a connection — an echo of the hope and dreams that had once filled the world. The music spoke of a time long past, of people who once laughed, loved, and dreamed. And even though Echo had never known such things, it couldn't help but feel the emptiness that came with their loss.
"""

# Simplify the text
simplified_text = model.generate_simplified_text(input_text)

# Output the simplified text
print("Simplified Text:")
print(simplified_text)

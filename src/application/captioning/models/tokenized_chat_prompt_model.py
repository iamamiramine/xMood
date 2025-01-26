import json

from langchain_core.prompts import PromptTemplate


class BaseTokenizedChatPromptTemplate:
    """
    Base class for tokenized chat prompt templates.
    Provides common functionality for loading prompt configurations and tokenization.
    """

    def __init__(self, tokenizer, input_variables, prompt_config_path="shared/config/prompts.json"):
        """
        Initialize the base prompt template.

        Args:
            input_variables: List of variables that will be used in the prompt template
            prompt_config_path: Path to JSON config file containing prompt templates
        """
        self.tokenizer = tokenizer
        self.input_variables = input_variables

        # Load prompt configurations from JSON file
        with open(prompt_config_path, "r") as f:
            self.prompt_config = json.load(f)

    def get_prompt(self):
        """
        Abstract method to get the formatted prompt template.
        Must be implemented by subclasses.

        Raises:
            NotImplementedError: If subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement get_prompt method")


class BasicTokenizedChatPromptTemplate(BaseTokenizedChatPromptTemplate):
    """
    A basic tokenized chat prompt template that handles simple question-answer interactions.
    Inherits from BaseTokenizedChatPromptTemplate to provide basic prompt functionality.
    """

    def __init__(self, tokenizer, prompt_config_path="shared/config/prompts.json"):
        """
        Initialize the basic prompt template.

        Args:
            prompt_config_path: Path to JSON config file containing prompt templates (default: "shared/config/prompts.json")
        """
        super().__init__(tokenizer, input_variables=["symbolic_features"], prompt_config_path=prompt_config_path)

    def get_prompt(self):
        """
        Constructs and returns a formatted prompt template for basic question-answer interactions.

        The prompt combines:
        1. A system base message from config
        2. A question placeholder
        3. An answer prefix for bash scripts

        Returns:
            PromptTemplate: A formatted template ready for question input
        """

        system = self.tokenizer.decode(
            self.tokenizer.encode(
                """system: You are a specialized music description system trained to generate natural language captions for musical pieces. Your task is to analyze and describe music based on detailed bar-level features and overall piece characteristics.
        You will ONLY generate a ONE Sentence Caption for each musical piece.
        
        Example:
        Bar-level Symbolic Features: [Bar_1, Time Signature_4/4, Key Signature_C:maj, Note Density_50, Mean Pitch_60, Mean Velocity_80, Mean Duration_5, Chord_C:maj7]
        Output Caption: A smooth and relaxing tune in C Major with a jazzy Cmaj7 vibe, steady rhythm, and warm, expressive notes that feel light and easy to follow.\n"""
            )
        )

        # Add input placeholders
        composer = self.tokenizer.decode(self.tokenizer.encode("""Composer: {composer}\n"""))
        genre = self.tokenizer.decode(self.tokenizer.encode("""Genre: {genre}\n"""))
        note_density = self.tokenizer.decode(self.tokenizer.encode("""Note Density: {note_density}\n"""))
        mean_velocity = self.tokenizer.decode(self.tokenizer.encode("""Mean Velocity: {mean_velocity}\n"""))
        mean_pitch = self.tokenizer.decode(self.tokenizer.encode("""Mean Pitch: {mean_pitch}\n"""))
        mean_duration = self.tokenizer.decode(self.tokenizer.encode("""Mean Duration: {mean_duration}\n"""))
        time_signature = self.tokenizer.decode(self.tokenizer.encode("""Time Signature: {time_signature}\n"""))
        key_signature = self.tokenizer.decode(self.tokenizer.encode("""Key Signature: {key_signature}\n"""))
        chords = self.tokenizer.decode(self.tokenizer.encode("""Chords: {chords}\n"""))
        instruments = self.tokenizer.decode(self.tokenizer.encode("""Instruments: {instruments}\n"""))
        emotions = self.tokenizer.decode(self.tokenizer.encode("""Emotions: {emotions}\n"""))
        answer = self.tokenizer.decode(self.tokenizer.encode("""Output Caption: \n"""))

        # Combine all parts
        full_prompt = (
            system
            + composer
            + genre
            + note_density
            + mean_velocity
            + mean_pitch
            + mean_duration
            + time_signature
            + key_signature
            + chords
            + instruments
            + emotions
            + answer
        )

        return PromptTemplate.from_template(full_prompt)

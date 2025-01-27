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
        
        Here are some examples to learn from:

        Example 1:
        Note Density: 2.1052631578947367
        Mean Pitch: 12.842105263157896
        Mean Velocity: 24.157894736842106
        Mean Duration: 30
        Time Signatures: ['Time Signature_4/4']
        Key Signatures: ['Key Signature_G#:minor']
        Chords: ['Chord_B:min', 'Chord_B:min7', 'Chord_D:maj', 'Chord_D:maj7', 'Chord_F#:None', 'Chord_N:N']
        Instruments: ['Instrument_Choir Aahs', 'Instrument_Electric Piano 2', 'Instrument_String Ensemble 1', 'Instrument_Synth Bass 2', 'Instrument_Synth Brass 1', 'Instrument_Tubular Bells', 'Instrument_drum']
        Output Caption: A hauntingly atmospheric piece in G# minor with a sparse 4/4 rhythm, blending minor and seventh chords like B:min7 and D:maj7, enriched by ethereal choir voices, tubular bells, and layered synth textures.

        Example 2:
        Note Density: 3.982142857142857
        Mean Pitch: 13.25
        Mean Velocity: 18.401785714285715
        Mean Duration: 25.214285714285715
        Time Signatures: ['Time Signature_4/4']
        Key Signatures: ['Key Signature_F:major']
        Chords: ['Chord_A:min7', 'Chord_B:min7', 'Chord_C:aug', 'Chord_C:dom7', 'Chord_C:maj', 'Chord_C:maj7', 'Chord_D:aug', 'Chord_D:dom7', 'Chord_E:min7', 'Chord_F#:maj7', 'Chord_F:None', 'Chord_F:maj', 'Chord_F:maj7', 'Chord_G:dom7', 'Chord_G:maj', 'Chord_G:maj7', 'Chord_G:min7', 'Chord_N:N']
        Instruments: ['Instrument_Acoustic Guitar (steel)', 'Instrument_Alto Sax', 'Instrument_Brass Section', 'Instrument_Distortion Guitar', 'Instrument_Electric Bass (finger)', 'Instrument_Electric Grand Piano', 'Instrument_Electric Guitar (clean)', 'Instrument_Electric Guitar (jazz)', 'Instrument_Electric Guitar (muted)', 'Instrument_String Ensemble 1', 'Instrument_Synth Choir', 'Instrument_drum']
        Output Caption: A mellow piece in F Major with a soft 4/4 rhythm, featuring a mix of jazzy and augmented chords like A:min7 and F:maj7, complemented by gentle acoustic guitar, electric piano, and a touch of brass and synth for a warm, layered texture.

        Now, analyze the following piece:
        """
            )
        )

        # Add input placeholders
        composer = self.tokenizer.decode(self.tokenizer.encode("""Composer: {composer}\n"""))
        genre = self.tokenizer.decode(self.tokenizer.encode("""Genre: {genre}\n"""))
        note_density = self.tokenizer.decode(self.tokenizer.encode("""Note Density: {note_density}\n"""))
        mean_velocity = self.tokenizer.decode(self.tokenizer.encode("""Mean Velocity: {mean_velocity}\n"""))
        mean_pitch = self.tokenizer.decode(self.tokenizer.encode("""Mean Pitch: {mean_pitch}\n"""))
        mean_duration = self.tokenizer.decode(self.tokenizer.encode("""Mean Duration: {mean_duration}\n"""))
        time_signature = self.tokenizer.decode(self.tokenizer.encode("""Time Signatures: {time_signature}\n"""))
        key_signature = self.tokenizer.decode(self.tokenizer.encode("""Key Signatures: {key_signature}\n"""))
        chords = self.tokenizer.decode(self.tokenizer.encode("""Chords: {chords}\n"""))
        instruments = self.tokenizer.decode(self.tokenizer.encode("""Instruments: {instruments}\n"""))
        # emotions = self.tokenizer.decode(self.tokenizer.encode("""Emotions: {emotions}\n"""))
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
            # + emotions
            + answer
        )

        return PromptTemplate.from_template(full_prompt)

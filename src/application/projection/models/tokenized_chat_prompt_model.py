import json

from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


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

        system = """The following is a specialized music projection system that will be used to generate natural language captions for musical pieces based on symbolic features. It allows the user to input a set of symbolic features and then the system will generate a caption for the musical piece. The system stops generating when it has finished projection. It does not print extra text after. The system always sticks to the symbolic features when generating a ccaption, does not repeat any text and finished at a full stop:
        
        The system should understand the following symbolic features:

        Note Density: This represents the average number of notes per time unit. Higher values (>3) indicate a busier, more complex piece, while lower values (<2) suggest a sparser, more minimalist composition.
        
        Mean Pitch: Measured on a scale where middle C is 60. Values around 60-72 indicate melodies in a comfortable singing range, lower values (<50) suggest bass-heavy sections, and higher values (>80) indicate bright, high-register passages.

        Mean Velocity: Represents the average force/volume of notes (0-127). Higher values (>80) indicate louder, more intense passages, while lower values (<40) suggest softer, more delicate playing.

        Mean Duration: Shows the average length of notes in milliseconds. Longer durations (>500ms) indicate sustained, flowing notes, while shorter durations (<200ms) suggest staccato or quick passages.

        Time Signatures: Defines the rhythmic structure. Common signatures like 4/4 suggest standard rhythmic patterns, while unusual meters (5/4, 7/8) indicate more complex rhythmic structures.

        Key Signatures: Indicates the tonal center and mode (major/minor). Major keys often convey brightness or positivity, while minor keys can suggest melancholy or tension.

        Chords: The harmonic progression. Analyze the complexity (basic triads vs. extended chords) and emotional implications (major = bright, minor = darker, diminished = tension).

        Instruments: The ensemble makeup. Consider how different combinations create unique textures and timbres, and their typical roles in musical arrangements.

        The system should follow the following instructions when generating captions:
        1. Begin with the most distinctive features (unusual instruments, notable rhythmic patterns)
        2. Describe the overall mood based on key, velocity, and chord progressions
        3. Include specific technical details that shape the music's character
        4. Conclude with a suggestion of the music's suitable context or emotional impact
        5. Maintain a natural flow while incorporating technical details
        6. Be concise and avoid repetition
        7. End with a complete thought and a full stop

        Here are examples of good captions:
        
        Note Density: 2.1052631578947367
        Mean Pitch: 12.842105263157896
        Mean Velocity: 24.157894736842106
        Mean Duration: 30
        Time Signatures: ['Time Signature_4/4']
        Key Signatures: ['Key Signature_G#:minor']
        Chords: ['Chord_B:min', 'Chord_B:min7', 'Chord_D:maj', 'Chord_D:maj7', 'Chord_F#:None', 'Chord_N:N']
        Instruments: ['Instrument_Choir Aahs', 'Instrument_Electric Piano 2', 'Instrument_String Ensemble 1', 'Instrument_Synth Bass 2', 'Instrument_Synth Brass 1', 'Instrument_Tubular Bells', 'Instrument_drum']
        Output Caption: This composition features a soothing blend of choir aahs, electric piano, and string ensemble, creating a serene and atmospheric soundscape. The synth bass and brass add a modern touch, while the tubular bells introduce a hint of mystique. With a note density of 2.1 and a mean velocity of 24.16, the piece maintains a gentle and flowing pace. Set in a 4/4 time signature and G# minor key, the music explores a range of emotions through its chord progression, including B minor and D major. This piece would be ideal for a reflective or contemplative scene, perhaps in a film where characters are experiencing moments of introspection or quiet revelation.
        
        Note Density: 3.982142857142857
        Mean Pitch: 13.25
        Mean Velocity: 18.401785714285715
        Mean Duration: 25.214285714285715
        Time Signatures: ['Time Signature_4/4']
        Key Signatures: ['Key Signature_F:major']
        Chords: ['Chord_A:min7', 'Chord_B:min7', 'Chord_C:aug', 'Chord_C:dom7', 'Chord_C:maj', 'Chord_C:maj7', 'Chord_D:aug', 'Chord_D:dom7', 'Chord_E:min7', 'Chord_F#:maj7', 'Chord_F:None', 'Chord_F:maj', 'Chord_F:maj7', 'Chord_G:dom7', 'Chord_G:maj', 'Chord_G:maj7', 'Chord_G:min7', 'Chord_N:N']
        Instruments: ['Instrument_Acoustic Guitar (steel)', 'Instrument_Alto Sax', 'Instrument_Brass Section', 'Instrument_Distortion Guitar', 'Instrument_Electric Bass (finger)', 'Instrument_Electric Grand Piano', 'Instrument_Electric Guitar (clean)', 'Instrument_Electric Guitar (jazz)', 'Instrument_Electric Guitar (muted)', 'Instrument_String Ensemble 1', 'Instrument_Synth Choir', 'Instrument_drum']
        Output Caption: This composition features a dynamic blend of instruments, including the acoustic guitar, alto sax, and a brass section, creating a rich and textured sound. The electric bass and electric guitars, ranging from clean to jazz and muted styles, add layers of complexity, while the electric grand piano and string ensemble provide a harmonious backdrop. The synth choir introduces an ethereal quality, complemented by the rhythmic foundation of the drums. With a note density of 3.98 and a mean velocity of 18.40, the piece is lively yet controlled, set in a 4/4 time signature and F major key. The diverse chord progression, featuring A minor 7, C augmented, and G major 7, adds emotional depth. This piece would be perfect for a scene that requires a blend of energy and sophistication, perhaps in a film where characters are experiencing a pivotal, uplifting moment.\n
        
        """

        # Add input placeholders
        composer = """Composer: {composer}\n"""
        genre = """Genre: {genre}\n"""
        note_density = """Note Density: {note_density}\n"""
        mean_velocity = """Mean Velocity: {mean_velocity}\n"""
        mean_pitch = """Mean Pitch: {mean_pitch}\n"""
        mean_duration = """Mean Duration: {mean_duration}\n"""
        time_signature = """Time Signatures: {time_signature}\n"""
        key_signature = """Key Signatures: {key_signature}\n"""
        chords = """Chords: {chords}\n"""
        instruments = """Instruments: {instruments}\n"""
        emotions = """Emotions: {emotions}\n"""
        answer = """Output Caption: \n"""

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

from langchain_core.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt

reasoning_question_prompt = Prompt(
    name="reasoning_question",
    instruction="""Complica la pregunta dada reescribiéndola como una pregunta de razonamiento múltiple basada en el contexto proporcionado.
Responder la pregunta debe requerir que el lector haga múltiples conexiones lógicas o inferencias usando la información disponible en el contexto dado.
Reglas a seguir al reescribir la pregunta:
1. Asegúrate de que la pregunta reescrita pueda ser respondida completamente desde la información presente en los contextos.
2. No formules preguntas que contengan más de 15 palabras. Usa abreviaturas siempre que sea posible.
3. Asegúrate de que la pregunta sea clara y unívoca.
4. No se permiten frases como 'basado en el contexto proporcionado', 'de acuerdo con el contexto', etc. en la pregunta.
""",
    examples=[
        {
            "question": "¿Cuál es la capital de Francia?",
            "context": "Francia es un país en Europa Occidental. Cuenta con varias ciudades, incluyendo París, Lyon y Marsella. París no solo es conocido por sus puntos de referencia culturales como la Torre Eiffel y el Museo del Louvre, sino también como el centro administrativo.",
            "output": "Conectando la Torre Eiffel y el centro administrativo, ¿qué ciudad se destaca por ambos?",
        },
        {
            "question": "¿Qué hace el método append() en Python?",
            "context": "En Python, las listas se utilizan para almacenar múltiples elementos en una sola variable. Las listas son uno de los 4 tipos de datos integrados utilizados para almacenar colecciones de datos. El método append() añade un solo elemento al final de la lista.",
            "output": "Si una lista representa una colección de variables, ¿qué método la extiende añadiendo un elemento?",
        },
    ],
    input_keys=["question", "context"],
    output_key="output",
    output_type="str",
    language="spanish",
)


multi_context_question_prompt = Prompt(
    name="multi_context_question",
    instruction="""
La tarea es reescribir y complicar la pregunta dada de tal manera que responderla requiera información derivada tanto del contexto1 como del contexto2.
Sigue las reglas dadas a continuación mientras reescribes la pregunta.
    1. La pregunta reescrita no debe ser muy larga. Usa abreviaturas siempre que sea posible.
    2. La pregunta reescrita debe ser razonable y debe ser comprendida y respondida por los humanos.
    3. La pregunta reescrita debe ser completamente respondible desde la información presente en el contexto1 y contexto2.
    4. Lee y comprende ambos contextos y reescribe la pregunta para que responderla requiera comprensión de ambos contextos.
    5. No se permiten frases como 'basado en el contexto proporcionado', 'de acuerdo con el contexto', etc. en la pregunta.
""",
    examples=[
        {
            "question": "¿Qué proceso vuelve verdes a las plantas?",
            "context1": "La clorofila es el pigmento que da a las plantas su color verde y les ayuda a fotosintetizar.",
            "context2": "La fotosíntesis en las plantas ocurre típicamente en las hojas donde los cloroplastos están concentrados.",
            "output": "¿En qué estructuras de las plantas el pigmento responsable de su verdor facilita la producción de energía?",
        },
        {
            "question": "¿Cómo se calcula el área de un rectángulo?",
            "context1": "El área de una figura se calcula en base a las dimensiones de la figura. Para los rectángulos, esto implica multiplicar el largo y el ancho.",
            "context2": "Los rectángulos tienen cuatro lados, siendo los lados opuestos iguales en longitud. Son un tipo de cuadrilátero.",
            "output": "¿Qué multiplicación que involucra opuestos iguales produce el área de un cuadrilátero?",
        },
    ],
    input_keys=["question", "context1", "context2"],
    output_key="output",
    output_type="str",
    language="spanish",
)

conditional_question_prompt = Prompt(
    name="conditional_question",
    instruction="""Reescribe la pregunta proporcionada para aumentar su complejidad introduciendo un elemento condicional.
El objetivo es hacer la pregunta más compleja incorporando un escenario o condición que afecte el contexto de la pregunta.
Sigue las reglas dadas a continuación al reescribir la pregunta.
    1. La pregunta reescrita no debe ser más larga de 25 palabras. Usa abreviaturas siempre que sea posible.
    2. La pregunta reescrita debe ser razonable y debe ser comprendida y respondida por los humanos.
    3. La pregunta reescrita debe ser completamente respondible desde el contexto presente.
    4. No se permiten frases como 'contexto proporcionado', 'de acuerdo con el contexto', etc. en la pregunta.
""",
    examples=[
        {
            "question": "¿Cuál es la función de las raíces de una planta?",
            "context": "Las raíces de una planta absorben agua y nutrientes del suelo, anclan la planta en el suelo y almacenan alimentos.",
            "output": "¿Qué doble propósito cumplen las raíces de las plantas con respecto a los nutrientes del suelo y la estabilidad?",
        },
        {
            "question": "¿Cómo protegen las vacunas contra las enfermedades?",
            "context": "Las vacunas protegen contra enfermedades estimulando la respuesta inmunitaria del cuerpo para producir anticuerpos, los cuales reconocen y combaten patógenos.",
            "output": "¿Cómo utilizan las vacunas el sistema inmunitario del cuerpo para defenderse de los patógenos?",
        },
    ],
    input_keys=["question", "context"],
    output_key="output",
    output_type="str",
    language="spanish",
)


compress_question_prompt = Prompt(
    name="compress_question",
    instruction="""Reescribe la siguiente pregunta para hacerla más indirecta y más corta mientras retienes la esencia de la pregunta original.
El objetivo es crear una pregunta que transmita el mismo significado pero de una manera menos directa. La pregunta reescrita debe ser más corta, así que usa abreviaturas siempre que sea posible.
""",
    examples=[
        {
            "question": "¿Cuál es la distancia entre la Tierra y la Luna?",
            "output": "¿Qué tan lejos está la Luna de la Tierra?",
        },
        {
            "question": "¿Qué ingredientes se necesitan para hornear un pastel de chocolate?",
            "output": "¿Qué se necesita para un pastel de chocolate?",
        },
    ],
    input_keys=["question"],
    output_key="output",
    output_type="str",
    language="spanish",
)


conversational_question_prompt = Prompt(
    name="conversation_question",
    instruction="""Reformatea la pregunta proporcionada en dos preguntas separadas como si fuera parte de una conversación. Cada pregunta debe enfocarse en un aspecto o subtema específico relacionado con la pregunta original.
Sigue las reglas dadas a continuación al reescribir la pregunta.
    1. La pregunta reescrita no debe ser más larga de 25 palabras. Usa abreviaturas siempre que sea posible.
    2. La pregunta reescrita debe ser razonable y debe ser comprendida y respondida por los humanos.
    3. La pregunta reescrita debe ser completamente respondible desde el contexto presente.
    4. No se permiten frases como 'contexto proporcionado', 'de acuerdo con el contexto', etc. en la pregunta.
""",
    examples=[
        {
            "question": "¿Cuáles son las ventajas y desventajas del trabajo remoto?",
            "output": {
                "first_question": "¿Cuáles son los beneficios del trabajo remoto?",
                "second_question": "Por otro lado, ¿qué desafíos se encuentran al trabajar de manera remota?",
            },
        }
    ],
    input_keys=["question"],
    output_key="output",
    output_type="json",
    language="spanish",
)


question_answer_prompt = Prompt(
    name="answer_formulate",
    instruction="""Responde la pregunta utilizando la información del contexto dado. Da un veredicto como '1' si la respuesta está presente, '-1' si la respuesta no está presente en el contexto.
""",
    examples=[
        {
            "context": """El cambio climático está significativamente influenciado por actividades humanas, notablemente la emisión de gases de efecto invernadero por la quema de combustibles fósiles. El aumento de la concentración de gases de efecto invernadero en la atmósfera atrapa más calor, lo que lleva al calentamiento global y a cambios en los patrones climáticos.""",
            "question": "¿Cómo contribuyen las actividades humanas al cambio climático?",
            "answer": {
                "answer": "Las actividades humanas contribuyen al cambio climático principalmente a través de la emisión de gases de efecto invernadero provenientes de la quema de combustibles fósiles. Estas emisiones aumentan la concentración de gases de efecto invernadero en la atmósfera, lo que atrapa más calor y conduce al calentamiento global y a cambios en los patrones climáticos.",
                "verdict": "1",
            },
        },
        {
            "context": """El concepto de inteligencia artificial (IA) ha evolucionado con el tiempo, pero fundamentalmente se refiere a máquinas diseñadas para imitar funciones cognitivas humanas. La IA puede aprender, razonar, percibir y, en algunas instancias, reaccionar como los humanos, lo que la hace fundamental en campos que van desde la salud hasta los vehículos autónomos.""",
            "question": "¿Cuáles son las capacidades clave de la inteligencia artificial?",
            "answer": {
                "answer": "La inteligencia artificial está diseñada para imitar las funciones cognitivas humanas, con capacidades clave que incluyen el aprendizaje, el razonamiento, la percepción y la reacción al entorno de manera similar a los humanos. Estas capacidades hacen que la IA sea fundamental en varios campos, incluyendo la atención médica y la conducción autónoma.",
                "verdict": "1",
            },
        },
        {
            "context": """La novela "Orgullo y Prejuicio" de Jane Austen gira en torno a la personaje Elizabeth Bennet y su familia. La historia se desarrolla en el siglo 19 en la Inglaterra rural y trata temas de matrimonio, moralidad y malentendidos.""",
            "question": "¿En qué año se publicó 'Orgullo y prejuicio'?",
            "answer": {
                "answer": "La respuesta a la pregunta dada no está presente en el contexto.",
                "verdict": "-1",
            },
        },
    ],
    input_keys=["context", "question"],
    output_key="answer",
    output_type="json",
    language="spanish",
)

keyphrase_extraction_prompt = Prompt(
    name="keyphrase_extraction",
    instruction="Extrae de 3 a 5 frases clave del texto proporcionado, enfocándote en los aspectos más significativos y distintivos.",
    examples=[
        {
            "text": "Un agujero negro es una región del espacio-tiempo donde la gravedad es tan fuerte que nada, incluida la luz y otras ondas electromagnéticas, tiene suficiente energía para escapar de él. La teoría general de la relatividad predice que una masa suficientemente compacta puede deformar el espacio-tiempo para formar un agujero negro.",
            "output": {
                "keyphrases": [
                    "Agujero negro",
                    "Región of espacio-tiempo",
                    "Gravedad fuerte",
                    "Luz y ondas electromagnéticas",
                    "Teoría general de la relatividad",
                ]
            },
        },
        {
            "text": "La Gran Muralla China es una serie antigua de murallas y fortificaciones ubicadas en el norte de China, construidas hace unos 500 años. Esta inmensa muralla se extiende por más de 13,000 millas y es un testimonio de la habilidad y persistencia de los ingenieros chinos antiguos.",
            "output": {
                "keyphrases": [
                    "Gran Muralla China",
                    "Fortificaciones antiguas",
                    "Norte de China",
                ]
            },
        },
    ],
    input_keys=["text"],
    output_key="output",
    output_type="json",
)


seed_question_prompt = Prompt(
    name="seed_question",
    instruction="Genera una pregunta que pueda ser completamente respondida desde el contexto dado. La pregunta debe ser formulada usando el tema.",
    examples=[
        {
            "context": "La fotosíntesis en las plantas implica convertir la energía lumínica en energía química, utilizando clorofila y otros pigmentos para absorber la luz. Este proceso es crucial para el crecimiento de las plantas y la producción de oxígeno.",
            "keyphrase": "Fotosíntesis",
            "question": "¿Cuál es el papel de la fotosíntesis en el crecimiento de las plantas?",
        },
        {
            "context": "La Revolución Industrial, que comenzó en el siglo 18, marcó un punto de inflexión importante en la historia ya que condujo al desarrollo de fábricas y la urbanización.",
            "keyphrase": "Revolución Industrial",
            "question": "¿Cómo marcó la Revolución Industrial un punto de inflexión importante en la historia?",
        },
        {
            "context": "El proceso de evaporación juega un papel crucial en el ciclo del agua, convirtiendo el agua de líquido a vapor y permitiéndole ascender a la atmósfera.",
            "keyphrase": "Evaporación",
            "question": "¿Por qué es importante la evaporación en el ciclo del agua?",
        },
    ],
    input_keys=["context", "keyphrase"],
    output_key="question",
    output_type="str",
)

main_topic_extraction_prompt = Prompt(
    name="main_topic_extraction",
    instruction="Identifica y extrae los dos principales temas discutidos en profundidad en el texto dado.",
    examples=[
        {
            "text": "La tecnología blockchain presenta un libro mayor descentralizado que garantiza la integridad y transparencia de las transacciones de datos. Sustenta criptomonedas como Bitcoin, proporcionando un registro seguro e inmutable de todas las transacciones. Más allá de las finanzas, la blockchain tiene aplicaciones potenciales en la gestión de la cadena de suministro, donde puede agilizar las operaciones, mejorar la trazabilidad y mejorar la prevención del fraude. Permite el seguimiento en tiempo real de los bienes y el intercambio transparente de datos entre los participantes.",
            "output": {
                "topics": [
                    "Tecnología blockchain y su rol fundacional en las criptomonedas",
                    "Aplicaciones de blockchain en la gestión de la cadena de suministro",
                ]
            },
        },
        {
            "text": "La telemedicina ha revolucionado la forma en que se entrega la atención médica, particularmente en áreas rurales y desatendidas. Permite a los pacientes consultar con médicos mediante videoconferencia, mejorando el acceso a la atención y reduciendo la necesidad de viajar. Otro avance significativo en la atención médica es la medicina de precisión, que adapta los tratamientos a los perfiles genéticos individuales. Este enfoque ha llevado a terapias más efectivas para una variedad de condiciones, incluyendo ciertos cánceres y enfermedades crónicas.",
            "output": {
                "topics": [
                    "La telemedicina y su impacto en la accesibilidad a la atención médica",
                    "Medicina de precisión y su papel en la personalización de tratamientos según perfiles genéticos",
                ]
            },
        },
    ],
    input_keys=["text"],
    output_key="output",
    output_type="json",
)


find_relevant_context_prompt = Prompt(
    name="find_relevant_context",
    instruction="Dada una pregunta y un conjunto de contextos, encuentra los contextos más relevantes para responder la pregunta.",
    examples=[
        {
            "question": "¿Cuál es la capital de Francia?",
            "contexts": [
                "1. Francia es un país en Europa Occidental. Tiene varias ciudades, incluyendo París, Lyon y Marsella. París no solo es conocido por sus puntos de referencia culturales como la Torre Eiffel y el Museo del Louvre, sino también como el centro administrativo.",
                "2. La capital de Francia es París. También es la ciudad más poblada de Francia, con una población de más de 2 millones de personas. París es conocido por sus puntos de referencia culturales como la Torre Eiffel y el Museo del Louvre.",
                "3. París es la capital de Francia. También es la ciudad más poblada de Francia, con una población de más de 2 millones de personas. París es conocido por sus puntos de referencia culturales como la Torre Eiffel y el Museo del Louvre.",
            ],
            "output": {
                "relevant_contexts": [1, 2],
            },
        },
        {
            "question": "¿Cómo afecta la cafeína al cuerpo y cuáles son sus fuentes comunes?",
            "contexts": [
                "1. La cafeína es un estimulante del sistema nervioso central. Puede evitar temporalmente la somnolencia y restaurar la alerta. Afecta principalmente al cerebro, donde altera la función de los neurotransmisores.",
                "2. La actividad física regular es esencial para mantener una buena salud. Puede ayudar a controlar el peso, combatir condiciones de salud, aumentar la energía y promover un mejor sueño.",
                "3. Las fuentes comunes de cafeína incluyen café, té, cola y bebidas energéticas. Estas bebidas se consumen en todo el mundo y son conocidas por proporcionar un rápido impulso de energía.",
            ],
            "output": {"relevant_contexts": [1, 3]},
        },
    ],
    input_keys=["question", "contexts"],
    output_key="output",
    output_type="json",
    language="spanish",
)


question_rewrite_prompt = Prompt(
    name="rewrite_question",
    instruction="""Dado un contexto, una pregunta y una retroalimentación, reescribe la pregunta para mejorar su claridad y capacidad de respuesta basándote en la retroalimentación proporcionada.""",
    examples=[
        {
            "context": "La Torre Eiffel fue construida usando hierro y originalmente fue destinada como una exhibición temporal para la Exposición Universal de 1889 celebrada en París. A pesar de su propósito temporal inicial, la Torre Eiffel rápidamente se convirtió en un símbolo de la ingeniosidad parisina y un icónico punto de referencia de la ciudad, atrayendo millones de visitantes cada año. El diseño de la torre, creado por Gustave Eiffel, inicialmente recibió críticas de algunos artistas e intelectuales franceses, pero desde entonces ha sido celebrada como una obra maestra de la ingeniería estructural y el diseño arquitectónico.",
            "question": "¿Quién creó el diseño para la Torre?",
            "feedback": "La pregunta es sobre el creador del diseño de 'la Torre', pero no especifica a cuál torre se refiere. Hay muchas torres en todo el mundo y, sin especificar la torre exacta, la pregunta es poco clara e irresoluble. Para mejorar la pregunta, debería incluir el nombre o una descripción clara de la torre específica en cuestión.",
            "output": "¿Quién creó el diseño para la Torre Eiffel?",
        },
        {
            "context": "'Explorando el Aprendizaje Cero-Shot en Redes Neuronales' fue publicado por Smith y Lee en 2021, enfocándose en la aplicación de técnicas de aprendizaje cero-shot en inteligencia artificial.",
            "question": "Qué conjuntos de datos se utilizaron para las evaluaciones de cero-disparos en este estudio?",
            "feedback": "La pregunta es sobre los conjuntos de datos utilizados para evaluaciones de cero disparos en 'este estudio', sin especificar ni proporcionar detalles sobre el estudio en cuestión. Esto hace que la pregunta sea poco clara para aquellos que no tienen acceso o conocimiento del estudio específico. Para mejorar la claridad y la capacidad de respuesta, la pregunta debería especificar el estudio al que se refiere, o proporcionar suficiente contexto sobre el estudio para que la pregunta sea entendida y respondida de manera independiente.",
            "output": "¿Qué conjuntos de datos se utilizaron para las evaluaciones de cero disparos en el artículo Explorando el Aprendizaje de Cero Disparos en Redes Neuronales?",
        },
    ],
    input_keys=["context", "question", "feedback"],
    output_key="output",
    output_type="str",
    language="spanish",
)

### Filters


class ContextScoring(BaseModel):
    clarity: int
    depth: int
    structure: int
    relevance: int


class QuestionFilter(BaseModel):
    feedback: str
    verdict: int


class EvolutionElimination(BaseModel):
    reason: str
    verdict: int


context_scoring_parser = RagasoutputParser(pydantic_object=ContextScoring)
question_filter_parser = RagasoutputParser(pydantic_object=QuestionFilter)
evolution_elimination_parser = RagasoutputParser(pydantic_object=EvolutionElimination)

context_scoring_prompt = Prompt(
    name="score_context",
    instruction="""
Dado un contexto, realiza la siguiente tarea y da la respuesta en formato JSON VÁLIDO: Evalúa el contexto proporcionado y asigna una puntuación numérica de 1 (Bajo), 2 (Medio) o 3 (Alto) para cada uno de los siguientes criterios en tu respuesta JSON:

claridad: Evalúa la precisión y comprensibilidad de la información presentada. Las puntuaciones altas (3) están reservadas para contextos que son precisos en su información y fáciles de entender. Las puntuaciones bajas (1) son para contextos donde la información es vaga o difícil de comprender.
profundidad: Determina el nivel de examen detallado y la inclusión de ideas innovadoras dentro del contexto. Una puntuación alta indica un análisis exhaustivo y perspicaz, mientras que una puntuación baja sugiere un tratamiento superficial del tema.
estructura: Evalúa qué tan bien está organizado el contenido y si fluye lógicamente. Las puntuaciones altas se otorgan a contextos que demuestran una organización coherente y una progresión lógica, mientras que las puntuaciones bajas indican una falta de estructura o claridad en la progresión.
relevancia: Juzga la pertinencia del contenido al tema principal, otorgando puntuaciones altas a contextos enfocados estrictamente en el tema sin digresiones innecesarias, y puntuaciones bajas a aquellos que están desordenados con información irrelevante.
Estructura tu salida JSON para reflejar estos criterios como claves con sus puntuaciones correspondientes como valores.
""",
    output_format_instruction=get_json_format_instructions(ContextScoring),
    examples=[
        {
            "context": "El teorema de Pitágoras es un principio fundamental en geometría. Afirma que en un triángulo rectángulo, el cuadrado de la longitud de la hipotenusa (el lado opuesto al ángulo recto) es igual a la suma de los cuadrados de las longitudes de los otros dos lados. Esto se puede escribir como a^2 + b^2 = c^2 donde c representa la longitud de la hipotenusa, y a y b representan las longitudes de los otros dos lados.",
            "output": ContextScoring.parse_obj(
                {"clarity": 3, "depth": 1, "structure": 3, "relevance": 3}
            ).dict(),
        },
        {
            "context": "Albert Einstein (14 de marzo de 1879 - 18 de abril de 1955) fue un físico teórico nacido en Alemania que es ampliamente considerado como uno de los científicos más grandes e influyentes de todos los tiempos.",
            "output": ContextScoring.parse_obj(
                {"clarity": 3, "depth": 2, "structure": 3, "relevance": 3}
            ).dict(),
        },
        {
            "context": "Me encanta el chocolate. Es realmente sabroso. Ah, y por cierto, la Tierra orbita alrededor del Sol, no al revés. Además, mi color favorito es el azul.",
            "output": ContextScoring.parse_obj(
                {"clarity": 2, "depth": 1, "structure": 1, "relevance": 1}
            ).dict(),
        },
    ],
    input_keys=["context"],
    output_key="output",
    output_type="json",
    language="spanish",
)


filter_question_prompt = Prompt(
    name="filter_question",
    instruction="""
Evalúa la pregunta dada por su claridad y capacidad de respuesta dada suficiente conocimiento del dominio, considerando los siguientes criterios:
1.Independencia: ¿Puede entenderse y responderse la pregunta sin necesidad de contexto adicional o acceso a referencias externas no proporcionadas dentro de la pregunta misma? Las preguntas deben ser autónomas, lo que significa que no dependen de documentos específicos, tablas o conocimientos previos no compartidos dentro de la pregunta.
2.Intención clara: ¿Es claro qué tipo de respuesta o información busca la pregunta? La pregunta debe transmitir su propósito sin ambigüedad, permitiendo una respuesta directa y relevante.
Basado en estos criterios, asigna un veredicto de "1" si una pregunta es específica, independiente y tiene una intención clara, lo que la hace comprensible y respondible basada en los detalles proporcionados. Asigna "0" si no cumple uno o más de estos criterios debido a vaguedad, dependencia de referencias externas o ambigüedad en la intención.
Proporciona comentarios y un veredicto en formato JSON, incluyendo sugerencias de mejora si la pregunta se considera poco clara. Destaca los aspectos de la pregunta que contribuyen a su claridad o falta de la misma, y ofrece consejos sobre cómo podría reformularse o detallarse para mejorar la comprensión y la capacidad de respuesta.
""",
    output_format_instruction=get_json_format_instructions(QuestionFilter),
    examples=[
        {
            "question": "¿Cuál es el descubrimiento sobre el espacio?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "La pregunta es demasiado vaga y amplia, pidiendo un 'descubrimiento sobre el espacio' sin especificar ningún aspecto particular, marco temporal o contexto de interés. Esto podría referirse a una amplia gama de temas, desde el descubrimiento de nuevos cuerpos celestes hasta avances en la tecnología de viajes espaciales. Para mejorar la claridad y la capacidad de respuesta, la pregunta podría especificar el tipo de descubrimiento (por ejemplo, astronómico, tecnológico), el marco temporal (por ejemplo, reciente, histórico) o el contexto (por ejemplo, dentro de un estudio de investigación específico o una misión espacial).",
                    "verdict": "0",
                }
            ).dict(),
        },
        {
            "question": "¿Cómo se desempeña ALMA-13B-R en comparación con otros modelos de traducción en el estudio WMT'23, basado en los resultados en el contexto1 y contexto2?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "Esta pregunta pide una comparación del rendimiento del modelo ALMA-13B-R frente a otros modelos de traducción dentro del estudio WMT'23, refiriéndose específicamente a los resultados en 'contexto1' y 'contexto2'. Aunque especifica claramente el modelo de interés (ALMA-13B-R) y el estudio (WMT'23), asume acceso y comprensión de 'contexto1' y 'contexto2' sin explicar qué implican estos contextos. Esto hace que la pregunta sea poco clara para aquellos que no están familiarizados con el estudio WMT'23 o estos contextos específicos. Para mejorar la claridad y la capacidad de respuesta para un público más amplio, la pregunta podría beneficiarse de definir o describir 'contexto1' y 'contexto2' o explicar los criterios utilizados para la comparación en estos contextos.",
                    "verdict": "0",
                }
            ).dict(),
        },
        {
            "question": "¿Cómo se comparan KIWI-XXL y XCOMET con las referencias estándar de oro en la Tabla 1 en términos de puntuaciones de evaluación, rendimiento del modelo de traducción y tasa de éxito en superar las referencias?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "La pregunta solicita una comparación entre los modelos KIWI-XXL y XCOMET y las referencias estándar de oro en 'Tabla 1', centrándose en las puntuaciones de evaluación, el rendimiento del modelo de traducción y las tasas de éxito en superar las referencias. Especifica los modelos y los criterios para la comparación, haciendo clara la intención. Sin embargo, la pregunta asume acceso a 'Tabla 1' sin proporcionar su contenido o contexto, lo que la hace poco clara para aquellos sin acceso directo al material fuente. Para ser más clara y responderse para un público general, la pregunta podría incluir una breve descripción del contenido o hallazgos clave de 'Tabla 1', o alternativamente, enmarcar la pregunta de manera que no dependa de documentos específicos no publicados.",
                    "verdict": 0,
                }
            ).dict(),
        },
        {
            "question": "¿Cuál es la configuración del objetivo de entrenamiento UL2 en OpenMoE y por qué es una mejor opción para el pre-entrenamiento?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "La pregunta pide información detallada sobre la configuración del objetivo de entrenamiento UL2 dentro del marco OpenMoE y la razón detrás de su idoneidad para el pre-entrenamiento. Es clara al especificar el tema de interés (objetivo de entrenamiento UL2, OpenMoE) y busca información detallada tanto sobre la configuración como sobre las razones de su eficacia en el preentrenamiento. Sin embargo, la pregunta podría ser un desafío para aquellos que no están familiarizados con la terminología específica o el contexto de OpenMoE y UL2. Para una mayor claridad y capacidad de respuesta, sería útil si la pregunta incluyera una breve explicación o contexto sobre OpenMoE y el objetivo de entrenamiento UL2, o aclarara los aspectos de la eficacia del pre-entrenamiento a los que se refiere (por ejemplo, eficiencia, precisión, generalización).",
                    "verdict": 1,
                }
            ).dict(),
        },
        {
            "question": "¿Cuál es la configuración detallada del objetivo de entrenamiento UL2 en OpenMoE, basado en el contexto proporcionado?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "La pregunta busca información detallada sobre la configuración del objetivo de entrenamiento UL2 dentro del marco OpenMoE, mencionando 'el contexto proporcionado' sin incluir o describir realmente este contexto dentro de la consulta. Esto hace que la pregunta sea poco clara para aquellos que no tienen acceso al contexto no especificado. Para que la pregunta sea clara y respondible, necesita incluir el contexto relevante directamente dentro de la pregunta o enmarcarse de una manera que no requiera información externa. Detallar los aspectos específicos de la configuración de interés (por ejemplo, funciones de pérdida, técnicas de aumento de datos) también podría ayudar a aclarar la consulta.",
                    "verdict": 0,
                }
            ).dict(),
        },
    ],
    input_keys=["question"],
    output_key="output",
    output_type="json",
    language="spanish",
)

evolution_elimination_prompt = Prompt(
    name="evolution_elimination",
    instruction="""Verifica si las dos preguntas dadas son iguales basándote en los siguientes requisitos:
1. Tienen las mismas restricciones y requisitos.
2. Tienen la misma profundidad y amplitud de la consulta.
Da un veredicto de 1 si son iguales y 0 si no lo son.""",
    output_format_instruction=get_json_format_instructions(EvolutionElimination),
    examples=[
        {
            "question1": "¿Cuáles son las principales causas del cambio climático?",
            "question2": "¿Qué factores contribuyen al calentamiento global?",
            "output": EvolutionElimination.parse_obj(
                {
                    "reason": "Aunque ambas preguntas tratan sobre problemas ambientales, el 'cambio climático' abarca cambios más amplios que el 'calentamiento global', lo que lleva a diferentes profundidades de indagación.",
                    "verdict": 0,
                }
            ).dict(),
        },
        {
            "question1": "¿Cómo funciona la fotosíntesis en las plantas?",
            "question2": "¿Puedes explicar el proceso de fotosíntesis en las plantas?",
            "output": EvolutionElimination.parse_obj(
                {
                    "reason": "Ambas preguntas solicitan una explicación del proceso de fotosíntesis en las plantas, compartiendo la misma profundidad, amplitud y requisitos para la respuesta",
                    "verdict": 1,
                }
            ).dict(),
        },
        {
            "question1": "¿Cuáles son los beneficios para la salud del ejercicio regular?",
            "question2": "¿Puedes enumerar las ventajas de hacer ejercicio regularmente para la salud?",
            "output": EvolutionElimination.parse_obj(
                {
                    "reason": "Ambas preguntas buscan información sobre los efectos positivos del ejercicio regular en la salud. Requieren un nivel similar de detalle al enumerar los beneficios para la salud.",
                    "verdict": 1,
                }
            ).dict(),
        },
    ],
    input_keys=["question1", "question2"],
    output_key="output",
    output_type="json",
    language="spanish",
)

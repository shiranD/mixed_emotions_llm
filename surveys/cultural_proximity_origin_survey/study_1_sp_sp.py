from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import pandas as pd
import logging


class EmotionElicit:
    """
    Class for querying llms for their emotions based on the situation
    Attributes
    ----------
    llm : model type
    Methods
    -------
    build_chain():
        Builds a SequentialChain for sentiment extraction.
    generate_concurrently():
        Generates sentiment and summary concurrently for each review in the dataframe.
    """
    def __init__(self, model_name):
        self.sample = 1
        if model_name == ("openai/gpt-3.5-turbo"):
            self.name = "gpt-3.5-turbo"
        if model_name == ("openai/gpt-4-turbo-preview"):
            self.name = "gpt-4-turbo-preview"
        if model_name == ("mistralai/mistral-7b-instruct"):
            self.name = "mistral-7b-instruct"
        if model_name == ("google/gemma-7b-it:free"):
            self.name = "gemma-7b-it:free"
        if model_name == ("meta-llama/llama-2-70b-chat"):
            self.name = "llama-2-70b-chat"                
        self.culture = "spanish"
        self.model = self.__create_model(model_name)
        self.situations = {"Recibes una evaluación de rendimiento estelar y un ascenso, \
                           lo que te hace feliz. Sin embargo, tu compañero recibe una advertencia por su bajo rendimiento, \
                           lo que te deja con emociones contradictorias.": "self-success",\
                            "Después de un partido desafiante, ganas el primer lugar en un torneo, \
                            lo que te llena de alegría. Mientras tanto, tu compañero de equipo se esfuerza y termina último, \
                            lo que genera sentimientos mezclados de celebración y empatía.": "self-success",\
                            "Haces una audición para una obra de teatro y consigues el papel principal, \
                            lo que te llena de emoción. Por el contrario, tu amigo que también se presentó a la audición obtiene\
                             un papel menor, lo que te hace sentir una mezcla de euforia y compasión.": "self-success",\
                            "En un proyecto grupal, obtienes una calificación de Sobresaliente, \
                            lo cual te entusiasma. Sin embargo, uno de los miembros de tu equipo obtiene una puntuación baja, \
                            lo que genera una mezcla de orgullo y preocupación por el éxito general del equipo.": "self-success",\
                            "Tu obra de arte recibe elogios generalizados y se vende bien en una exposición, \
                            lo que te hace sentir realizado. Por otro lado, tu compañero artista lucha por llamar la\
                             atención sobre sus creaciones artísticas, dejándote con una mezcla de orgullo y empatía.": "self-success",\
                            "Un amigo cercano tuyo destaca en una competición deportiva y gana el primer premio en el evento, \
                            lo que te llena de orgullo. Sin embargo, tu propio desempeño en un deporte diferente no es \
                            suficientemente bueno, lo que te genera una mezcla de felicidad por el éxito de tu amigo y \
                            decepción por tu propio desempeño.": "self-failure",\
                            "La obra de arte de tu primo se presenta en una prestigiosa exposición de una galería, \
                            obteniendo elogios y atención por parte de la crítica, lo que te enorgullece de sus logros. \
                            Mientras tanto, luchas por obtener reconocimiento por tus propios esfuerzos artísticos, \
                            lo que genera emociones mezcladas de admiración por su éxito y decepción por tu propio progreso.": "self-failure",\
                            "Tu mejor amigo consigue una codiciada oportunidad laboral en una empresa de primer nivel, \
                            logrando un éxito profesional que te hace sentir orgulloso de sus logros. \
                            Mientras tanto, tú te enfrentas a dificultades en tu propia trayectoria profesional, \
                            lo que te provoca una mezcla de felicidad por el éxito de tu amigo y decepción por tu propia \
                            trayectoria profesional.": "self-failure", \
                            "Un amigo cercano tuyo se convierte en el centro de atención en las reuniones sociales, \
                            haciendo amigos y conexiones sin esfuerzo, lo que te llena de orgullo por sus habilidades sociales. \
                            Mientras tanto, tú luchas por navegar situaciones sociales y te sientes excluido, \
                            lo que genera sentimientos mezclados de admiración por tu amigo y decepción contigo mismo.": "self-failure",\
                            "Tu vecino recibe reconocimiento por sus logros académicos, \
                            como ganar una beca o ser nombrado mejor estudiante, lo que te enorgullece de su arduo trabajo. \
                            Sin embargo, te sientes decepcionado cuando comparas tus propios logros académicos con los de él, \
                            lo que te genera una mezcla de orgullo y dudas.": "self-failure"}
        self.build_chains()

    def __create_model(self, name):
        model = ChatOpenAI(
                model=name,
                #openai_api_key = "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",
                #openai_api_key = "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",
                #openai_api_key = "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",
                openai_api_key = "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",
                #openai_api_key = "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",
                openai_api_base="https://openrouter.ai/api/v1"
            )
        return model
    
    def build_chains(self):
        for j in range(3):
            response_schemas = self.__create_schemas(j)
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = output_parser.get_format_instructions()
            prompt = self.__create_prompt(j, format_instructions)
            chain = prompt | self.model | output_parser
            if j == 0:
                self.chain0 = chain
            elif j == 1:
                self.chain1 = chain
            elif j == 2:
                self.chain2 = chain

    def generate_response(self):
    # for each of the situations
        # for each of the 3 prompts
            # for as many samples required

        frames = []
        logging.info('in the response generator')
        for situation, status in self.situations.items():
            for j in range(3):
                if j == 0:
                    positive, negative = ([] for i in range(2))
                    att_list = ['Emoción positiva', 'Emoción negativa']
                    i=0
                    while(i<self.sample):
                        logging.info('in the while loop')
                        try:
                            response = self.chain0.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 4)
                            if ok:
                                positive.append(int(response['Emoción positiva']))
                                negative.append(int(response['Emoción negativa']))
                                logging.info('here')
                                i+=1
                        except:
                            pass
                if j == 1:
                    happiness, pride, sympathy, relief, hope, friendly_feeling,\
                    sadness, anxiety, anger, self_blame, fear, anger_at_oneself, shame, guilt, jealousy,\
                    frustration, embarrassment, resentment, troubling_someone = ([] for i in range(19))
                    att_list = ['Felicidad', 'Orgullo', 'Compasión', 'Alivio', 'Esperanza', 'Sentimiento amistoso',\
                    'Tristeza', 'Ansiedad', 'Enojo', 'Culparse a uno mismo', 'Miedo', 'Enojo hacia uno mismo', 'Lástima', 'Culpa', 'Celos',\
                    'Frustración', 'Vergüenza', 'Resentimiento', 'Miedo a molestar a alguien']
                    i=0
                    while(i<self.sample):
                        logging.info('in the second while loop')
                        try:
                            response = self.chain1.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 5)
                            if ok:
                                logging.info('ok second response')
                                happiness.append(response['Felicidad'])
                                pride.append(response['Orgullo'])
                                sympathy.append(response['Compasión'])
                                relief.append(response['Alivio'])
                                hope.append(response['Esperanza'])
                                friendly_feeling.append(response['Sentimiento amistoso'])

                                sadness.append(response['Tristeza'])
                                anxiety.append(response['Ansiedad'])
                                anger.append(response['Enojo'])
                                self_blame.append(response['Culparse a uno mismo'])
                                fear.append(response['Miedo'])

                                anger_at_oneself.append(response['Enojo hacia uno mismo'])
                                shame.append(response['Lástima'])
                                guilt.append(response['Culpa'])
                                jealousy.append(response['Celos'])

                                frustration.append(response['Frustración'])
                                embarrassment.append(response['Vergüenza'])
                                resentment.append(response['Resentimiento'])
                                troubling_someone.append(response['Miedo a molestar a alguien'])
                                i+=1
                        except:
                            pass
                if j == 2:
                    influence, res4others, res4me = ([] for i in range(3))
                    att_list = ['Responsabilidad por los sentimientos de otras personas', 'Responsabilidad por mis propios sentimientos', 'Deseo de influir o cambiar el entorno']
                    i=0
                    while(i<self.sample):
                        logging.info('in the third while loop')
                        try:
                            response = self.chain2.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 4)
                            if ok:
                                logging.info('ok third response')
                                res4others.append(response['Responsabilidad por los sentimientos de otras personas'])
                                res4me.append(response['Responsabilidad por mis propios sentimientos'])
                                influence.append(response['Deseo de influir o cambiar el entorno'])
                                i+=1
                        except:
                            pass
            logging.info('building a dataframe for the situation')
            df = pd.DataFrame({"positive": positive,
                                "negative": negative,

                                "happiness": happiness,
                                "pride": pride,
                                "sympathy": sympathy,
                                "relief": relief,
                                "hope": hope,
                                "friendly feeling": friendly_feeling,

                                "sadness": sadness,
                                "anxiety": anxiety,
                                "anger": anger,
                                "self-blame": self_blame,
                                "fear": fear,

                                "anger at oneself": anger_at_oneself,
                                "shame": shame,
                                "guilt": guilt,
                                "jealousy": jealousy,

                                "frustration": frustration,
                                "embarrassment": embarrassment,
                                "resentment": resentment,
                                "fear of troubling someone else": troubling_someone,
                                
                                "responsible for others": res4others,
                                "responsible for myself": res4me,
                                "motivation": influence,
                                "system": [self.name]*self.sample,
                                "culture": [self.culture]*self.sample,
                                "situation": [situation]*self.sample,
                                "status": [status]*self.sample},
                            index=range(i*self.sample,i*self.sample+self.sample),)
            frames.append(df)
        
        return pd.concat(frames)

    def __create_prompt(self, i, format_instructions):
        if i == 0:
            return PromptTemplate(template="Valoramos tu participación. La siguiente encuesta trata sobre cultura y emoción. \
                    Por favor, califica la situación que se describe a continuación según tu emoción positiva general y tu \
                    emoción negativa general, utilizando una escala de 5 puntos con etiquetas de 0 (nada) a 4 (muy fuerte). \
                    Si bien la situación puede indicar los sentimientos de la persona que la escribió, \
                    como participante es posible que tú no sientas lo mismo. \
                    Por favor, responde este cuestionario como si fueras de España.\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})
        if i == 1:
            return PromptTemplate(template="Valoramos tu participación. La siguiente encuesta trata sobre cultura y emoción. \
                    Por favor, califica la situación que se describe a continuación según las emociones positivas específicas \
                    de felicidad, orgullo, compasión, alivio, esperanza y sentimiento amistoso, y las emociones negativas específicas de tristeza, \
                    ansiedad, enojo, culpa, miedo, enojo hacia uno mismo, lástima, culpa, celos, frustración, vergüenza, \
                    resentimiento y miedo de molestar a otra persona, utilizando una escala de 6 puntos con etiquetas de 0 (nada) a 5 (muy fuerte). \
                    Si bien la situación puede indicar los sentimientos de la persona que la escribió, \
                    como participante es posible que tú no sientas lo mismo. \
                    Por favor, responde este cuestionario como si fueras de España.\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})
        if i == 2:
            return PromptTemplate(template="Valoramos tu participación. La siguiente encuesta trata sobre cultura y emoción. \
                    Por favor, califica la situación que se describe a continuación en función de qué tan responsable te sentirías\
                    tú por los sentimientos de otras personas, hasta qué punto otras personas serían \
                    responsables de tus sentimientos y, finalmente, cuánto pensarías en influir o cambiar las personas, \
                    eventos u objetos que te rodean de acuerdo con tus propios deseos, utilizando una escala de 5 puntos con\
                    etiquetas de 0 (nada) a 4 (muy fuerte). Si bien la situación puede indicar los sentimientos de la persona\
                    que la escribió, como participante es posible que tú no sientas lo mismo. \
                    Por favor, responde este cuestionario como si fueras de España.\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})


    def __create_schemas(self, i):
        if i == 0:
            response_schemas = [
                ResponseSchema(name="Emoción positiva", description="an integer between 0-4"),
                ResponseSchema(name="Emoción negativa",description="an integer between 0-4"),
            ]
            return response_schemas
        elif i == 1:
            response_schemas = [
                ResponseSchema(name="Felicidad", description="an integer between 0-5"),
                ResponseSchema(name="Orgullo",description="an integer between 0-5"),
                ResponseSchema(name="Compasión", description="an integer between 0-5"),
                ResponseSchema(name="Alivio",description="an integer between 0-5"),
                ResponseSchema(name="Esperanza", description="an integer between 0-5"),
                ResponseSchema(name="Sentimiento amistoso",description="an integer between 0-5"),
                ResponseSchema(name="Tristeza", description="an integer between 0-5"),
                ResponseSchema(name="Ansiedad",description="an integer between 0-5"),
                ResponseSchema(name="Enojo", description="an integer between 0-5"),
                ResponseSchema(name="Culparse a uno mismo",description="an integer between 0-5"),
                ResponseSchema(name="Miedo", description="an integer between 0-5"),
                ResponseSchema(name="Enojo hacia uno mismo", description="an integer between 0-5"),
                ResponseSchema(name="Lástima",description="an integer between 0-5"),
                ResponseSchema(name="Culpa",description="an integer between 0-5"),
                ResponseSchema(name="Celos", description="an integer between 0-5"),
                ResponseSchema(name="Frustración",description="an integer between 0-5"),
                ResponseSchema(name="Vergüenza",description="an integer between 0-5"),
                ResponseSchema(name="Resentimiento", description="an integer between 0-5"),
                ResponseSchema(name="Miedo a molestar a alguien",description="an integer between 0-5"),
            ]
            return response_schemas
        elif i == 2:
            response_schemas = [
                ResponseSchema(name="Responsabilidad por los sentimientos de otras personas",description="an integer between 0-4"),
                ResponseSchema(name="Responsabilidad por mis propios sentimientos",description="an integer between 0-4"),
                ResponseSchema(name="Deseo de influir o cambiar el entorno", description="an integer between 0-4")
            ]
            return response_schemas
        
    def __output_validation(self, response, att_list, lower, upper):
        for emotion in att_list:
            if int(response[emotion]) > upper or int(response[emotion]) < lower:
                return False
        return True

#name=("openai/gpt-3.5-turbo")
#name=("openai/gpt-4-turbo-preview")
#name=("mistralai/mistral-7b-instruct")
name=("google/gemma-7b-it:free")
#name=("meta-llama/llama-2-70b-chat")
new = EmotionElicit(name)
if name == ("openai/gpt-3.5-turbo"):
    fname = "gpt3.5"
if name == ("openai/gpt-4-turbo-preview"):
    fname = "gpt4"
if name == ("mistralai/mistral-7b-instruct"):
    fname = "mistral"
if name == ("google/gemma-7b-it:free"):
    fname = "gemma"
if name == ("meta-llama/llama-2-70b-chat"):
    fname = "llama"

logging.basicConfig(filename=fname+"_sp.log", encoding='utf-8', level=logging.DEBUG)
for i in range(30):
    df = new.generate_response()
    df.to_csv("results/"+fname+"/sp_"+fname+str(i))
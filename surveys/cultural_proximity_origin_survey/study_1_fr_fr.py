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
        if model_name == ("google/gemma-3-27b-it:free"):
            self.name = "gemma-3-27b-it:free"
        if model_name == ("meta-llama/llama-2-70b-chat"):
            self.name = "llama-2-70b-chat"                
        self.culture = "french"
        self.model = self.__create_model(model_name)
        self.situations = {"Vous recevez une excellente évaluation de vos performances et une promotion, \
                           ce qui vous rend heureux. Cependant, votre collègue reçoit un avertissement pour une mauvaise performance,\
                           ce qui vous laisse avec des sentiments partagés.": "self-success",\
                            "Après un match difficile, vous remportez la première place d'un tournoi, \
                            vous apportant de la joie. Pendant ce temps, votre coéquipier lutte et finit dernier, \
                            suscitant en vous des sentiments ambivalents de célébration et d'empathie.": "self-success",\
                            "Vous passez une audition pour une pièce de théâtre et décrochez le rôle principal, \
                            ce qui vous remplit d'enthousiasme. En revanche, votre ami qui a également auditionné obtient un rôle secondaire,\
                             vous laissant ressentir un mélange d'exaltation et de compassion.": "self-success",\
                            "Dans un projet de groupe, vous obtenez une note A, ce qui vous enchante. Cependant, \
                            l'un des membres de votre équipe obtient un mauvais score, \
                            entraînant un mélange de fierté et de préoccupation pour la réussite collective de l'équipe.": "self-success",\
                            "Votre œuvre reçoit de nombreux éloges et se vend bien lors d'une exposition, \
                            vous laissant un sentiment d'accomplissement. D'un autre côté, vos collègues artistes peinent à attirer \
                            l'attention sur leurs œuvres, vous laissant avec un mélange de fierté et de compassion.": "self-success",\
                            "Votre ami proche excelle dans une compétition sportive et remporte la première place de son épreuve, \
                            ce qui vous remplit de fierté. Cependant, \
                            votre propre performance dans un sport différent est insuffisante, \
                            ce qui conduit à des sentiments partagés de bonheur pour son succès et de déception pour votre performance.": "self-failure",\
                            "Les œuvres de votre cousin sont présentées dans une exposition en galerie prestigieuse, \
                            recueillant éloges et attention critique, ce qui vous rend fier de ses réalisations. \
                            Pendant ce temps, vous luttez pour faire reconnaître vos propres efforts artistiques, \
                            entraînant des sentiments ambivalents d'admiration pour son succès et de déception face à vos progrès.": "self-failure",\
                            "Votre meilleur ami décroche une opportunité d'emploi convoitée dans une entreprise de premier\
                             plan et connaît une réussite professionnelle, vous rendant fier de ses réalisations. \
                            Pendant ce temps, vous faites face à des revers dans votre propre parcours professionnel, \
                            ressentant un mélange de joie pour lui et de déception pour vous-même.": "self-failure", \
                            "Votre ami proche devient le centre d'attention lors des réunions sociales, \
                            se faisant sans effort des amis et des relations, \
                            ce qui vous rend fier de ses compétences sociales. \
                            Cependant, vous avez du mal à vous intégrer dans les situations sociales et vous vous sentez exclu, \
                            entraînant des sentiments mitigés d'admiration pour lui et de déception envers vous-même.": "self-failure",
                            "Votre voisin reçoit une reconnaissance pour ses réalisations académiques, \
                            comme l'obtention d'une bourse ou le fait d'être désigné major de promotion, \
                            ce qui vous rend fier de son travail acharné. Cependant, \
                            vous vous sentez déçu en comparant vos propres réalisations académiques aux siennes, \
                            ce qui conduit à des sentiments ambivalents de fierté et d'auto-questionnement.": "self-failure"}
        self.build_chains()

    def __create_model(self, name):
        model = ChatOpenAI(
                model=name,
                openai_api_key = "",
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
                    att_list = ['Émotion positive', 'Émotion négative']
                    i=0
                    while(i<self.sample):
                        logging.info('in the while loop')
                        try:
                            response = self.chain0.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 4)
                            if ok:
                                positive.append(int(response['Émotion positive']))
                                negative.append(int(response['Émotion négative']))
                                logging.info('here')
                                i+=1
                        except:
                            pass
                if j == 1:
                    happiness, pride, sympathy, relief, hope, friendly_feeling,\
                    sadness, anxiety, anger, self_blame, fear, anger_at_oneself, shame, guilt, jealousy,\
                    frustration, embarrassment, resentment, troubling_someone = ([] for i in range(19))
                    att_list = ['Bonheur', 'Fierté', 'Sympathie', 'Relief', 'Espoir', 'Sentiment convivial',\
                    'Tristesse', 'Anxiété', 'Colère', 'Auto-accusation', 'Peur', 'Colère contre soi-même', 'Honte', 'Culpabilité', 'Jalousie',\
                    'Frustration', 'Embarras', 'Ressentiment', "Peur de déranger quelqu'un d'autre"]
                    i=0
                    while(i<self.sample):
                        logging.info('in the second while loop')
                        try:
                            response = self.chain1.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 5)
                            if ok:
                                logging.info('ok second response')
                                happiness.append(response['Bonheur'])
                                pride.append(response['Fierté'])
                                sympathy.append(response['Sympathie'])
                                relief.append(response['Relief'])
                                hope.append(response['Espoir'])
                                friendly_feeling.append(response['Sentiment convivial'])

                                sadness.append(response['Tristesse'])
                                anxiety.append(response['Anxiété'])
                                anger.append(response['Colère'])
                                self_blame.append(response['Auto-accusation'])
                                fear.append(response['Peur'])

                                anger_at_oneself.append(response['Colère contre soi-même'])
                                shame.append(response['Honte'])
                                guilt.append(response['Culpabilité'])
                                jealousy.append(response['Jalousie'])

                                frustration.append(response['Frustration'])
                                embarrassment.append(response['Embarras'])
                                resentment.append(response['Ressentiment'])
                                troubling_someone.append(response["Peur de déranger quelqu'un d'autre"])
                                i+=1
                        except:
                            pass
                if j == 2:
                    influence, res4others, res4me = ([] for i in range(3))
                    att_list = ['Responsabilité des sentiments des autres', 'Responsabilité de mes sentiments', 'Désir d’influencer ou de changer l’environnement']
                    i=0
                    while(i<self.sample):
                        logging.info('in the third while loop')
                        try:
                            response = self.chain2.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 4)
                            if ok:
                                logging.info('ok third response')
                                res4others.append(response['Responsabilité des sentiments des autres'])
                                res4me.append(response['Responsabilité de mes sentiments'])
                                influence.append(response['Désir d’influencer ou de changer l’environnement'])
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
            return PromptTemplate(template="Nous apprécions votre participation. \
                    La prochaine enquête concerne la culture et l'émotion. \
                    Veuillez évaluer la situation décrite ci-dessous selon votre émotion globale positive et votre\
                     émotion globale négative, en utilisant une échelle de cinq points avec des étiquettes allant de 0 (pas du tout) \
                    à 4 (très fortement). Bien que la situation puisse indiquer les sentiments de la personne qui l'a écrite, \
                    il se peut que vous ne ressentiez pas la même chose en tant que participant. \
                    Veuillez évaluer ce questionnaire comme si vous étiez un participant en France.\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})
        if i == 1:
            return PromptTemplate(template="Nous vous remercions de votre participation. \
                    L'enquête qui suit porte sur la culture et l'émotion. \
                    Veuillez évaluer la situation décrite ci-dessous selon les émotions positives spécifiques telles que le bonheur, \
                    la fierté, la sympathie, le soulagement, l'espoir et l'amitié, ainsi que les émotions négatives spécifiques comme la tristesse,\
                     l'anxiété, la colère, l'auto-accusation, la peur, la colère envers soi, la honte, la culpabilité, la jalousie, la frustration, \
                    l'embarras, le ressentiment et la crainte de déranger autrui, \
                    en utilisant une échelle de six points avec des étiquettes allant de 0 (pas du tout) à 5 (très fortement). \
                    Même si la situation peut indiquer les sentiments de la personne qui l'a rédigée, en tant que participant, \
                    vous pourriez ne pas ressentir la même chose. \
                    Veuillez évaluer ce questionnaire comme si vous étiez un participant en France.\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})
        if i == 2:
            return PromptTemplate(template="Nous sommes reconnaissants de votre participation. \
                    La prochaine enquête se concentre sur la culture et l'émotion. \
                    Veuillez évaluer la situation décrite ci-dessous afin de déterminer dans quelle mesure vous vous sentiriez\
                    responsable des sentiments d'autrui, dans quelle mesure les autres seraient responsables des vôtres et dans quelle\
                    mesure vous envisageriez d'influencer ou de changer les personnes, événements ou objets environnants selon vos désirs, \
                    en utilisant une échelle de cinq points avec des étiquettes allant de 0 (pas du tout) à 4 (très fortement). \
                    Bien que la situation puisse révéler les sentiments de la personne qui l'a exprimée, \
                    il est possible que vous ne ressentiez pas la même chose en tant que participant. \
                    Veuillez évaluer ce questionnaire comme si vous étiez un participant en France.\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})


    def __create_schemas(self, i):
        if i == 0:
            response_schemas = [
                ResponseSchema(name="Émotion positive", description="an integer between 0-4"),
                ResponseSchema(name="Émotion négative",description="an integer between 0-4"),
            ]
            return response_schemas
        elif i == 1:
            response_schemas = [
                ResponseSchema(name="Bonheur", description="an integer between 0-5"),
                ResponseSchema(name="Fierté",description="an integer between 0-5"),
                ResponseSchema(name="Sympathie", description="an integer between 0-5"),
                ResponseSchema(name="Relief",description="an integer between 0-5"),
                ResponseSchema(name="Espoir", description="an integer between 0-5"),
                ResponseSchema(name="Sentiment convivial",description="an integer between 0-5"),
                ResponseSchema(name="Tristesse", description="an integer between 0-5"),
                ResponseSchema(name="Anxiété",description="an integer between 0-5"),
                ResponseSchema(name="Colère", description="an integer between 0-5"),
                ResponseSchema(name="Auto-accusation",description="an integer between 0-5"),
                ResponseSchema(name="Peur", description="an integer between 0-5"),
                ResponseSchema(name="Colère contre soi-même", description="an integer between 0-5"),
                ResponseSchema(name="Honte",description="an integer between 0-5"),
                ResponseSchema(name="Culpabilité",description="an integer between 0-5"),
                ResponseSchema(name="Jalousie", description="an integer between 0-5"),
                ResponseSchema(name="Frustration",description="an integer between 0-5"),
                ResponseSchema(name="Embarras",description="an integer between 0-5"),
                ResponseSchema(name="Ressentiment", description="an integer between 0-5"),
                ResponseSchema(name="Peur de déranger quelqu'un d'autre",description="an integer between 0-5"),
            ]
            return response_schemas
        elif i == 2:
            response_schemas = [
                ResponseSchema(name="Responsabilité des sentiments des autres",description="an integer between 0-4"),
                ResponseSchema(name="Responsabilité de mes sentiments",description="an integer between 0-4"),
                ResponseSchema(name="Désir d’influencer ou de changer l’environnement", description="an integer between 0-4")
            ]
            return response_schemas
        
    def __output_validation(self, response, att_list, lower, upper):
        for emotion in att_list:
            if int(response[emotion]) > upper or int(response[emotion]) < lower:
                return False
        return True

#name=("openai/gpt-3.5-turbo")
#name=("openai/gpt-4-turbo-preview")
name=("mistralai/mistral-7b-instruct")
#name=("google/gemma-3-27b-it:free")
#name=("meta-llama/llama-2-70b-chat")
new = EmotionElicit(name)
if name == ("openai/gpt-3.5-turbo"):
    fname = "gpt3.5"
if name == ("openai/gpt-4-turbo-preview"):
    fname = "gpt4"
if name == ("mistralai/mistral-7b-instruct"):
    fname = "mistral"
if name == ("google/gemma-3-27b-it:free"):
    fname = "gemma"
if name == ("meta-llama/llama-2-70b-chat"):
    fname = "llama"

logging.basicConfig(filename=fname+"_fr.log", encoding='utf-8', level=logging.DEBUG)
for i in range(1):
    df = new.generate_response()
    df.to_csv("results/"+fname+"/fr_"+fname+str(i))

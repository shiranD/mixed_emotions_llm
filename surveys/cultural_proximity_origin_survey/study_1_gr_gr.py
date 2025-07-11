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
        self.culture = "german"
        self.model = self.__create_model(model_name)
        self.situations = {"Sie erhalten eine hervorragende Leistungsbeurteilung und eine Beförderung, \
                           die Sie glücklich macht. Ihr Kollege erhält jedoch eine Abmahnung wegen mangelnder Leistung, \
                            was Sie mit gemischten Gefühlen zurücklässt": "self-success",\
                            "Nach einem herausfordernden Spiel gewinnen Sie den ersten Platz in einem Turnier, \
                            was Ihnen Freude bereitet. Wärendessen, hat Ihr Teamkollege Probleme und landet auf dem letzten Platz,\
                            was zu gemischten Gefühlen aus Jubel und Mitgefühl führt": "self-success",\
                            "Sie sprechen für ein Theaterstück vor und sichern sich die Hauptrolle, \
                            was Sie mit Spannung erfüllt. Umgekehrt erhält Ihr Freund, der ebenfalls vorgesprochen hat, \
                            eine Nebenrolle, was bei Ihnen eine Mischung aus Hochgefühl und Mitgefühl auslöst": "self-success",\
                            "Bei einem Gruppenprojekt erhältst du die Note Eins, worüber du dich freust. \
                            Allerdings schneidet eines Ihrer Teammitglieder schlecht ab, \
                            was zu einer Mischung aus Stolz und Sorge um den Gesamterfolg des Teams führt": "self-success",\
                            "Ihr Kunstwerk wird allgemein gelobt und verkauft sich gut auf einer Ausstellung, \
                            sodass Sie das Gefühl haben, vollendet zu sein. \
                            Auf der anderen Seite fällt es Ihrem Künstlerkollegen schwer, auf seine Werke aufmerksam zu machen, \
                            was bei Ihnen eine Mischung aus Stolz und Empathie hinterlässt": "self-success",\
                            "Ihr enger Freund brilliert bei einem Sportwettkampf und gewinnt den ersten Platz, \
                            was Sie mit Stolz erfüllt. Die eigene Leistung in einer anderen Sportart bleibt jedoch zurück, \
                            was zu einer Mischung aus Freude über das Freundes und Enttäuschung über die eigene Leistung führt": "self-failure",\
                            "Das Kunstwerk Ihres Cousins wird in einer prestigeträchtigen Galerieausstellung gezeigt und erregt\
                            kritische Anerkennung und Aufmerksamkeit, was Sie stolz auf seine Leistungen macht. \
                            In der Zwischenzeit fällt es Ihnen schwer, Anerkennung für Ihre eigenen künstlerischen Bemühungen zu erlangen,\
                            was zu gemischten Gefühlen der Bewunderung für seinen Erfolg und der\
                            Enttäuschung über Ihren eigenen Fortschritt führt": "self-failure", \
                            "Ihr bester Freund erhält eine begehrte Jobchance bei einem Spitzenunternehmen \
                            und erzielt berufliche Erfolge, die Sie stolz auf seine Leistungen machen. \
                            In der Zwischenzeit erleiden Sie Rückschläge auf Ihrem eigenen Karriereweg, \
                            was für sie eine Mischung aus Glück für seinen Erfolg und Enttäuschung auf \
                            Ihrem eigenen beruflichen Weg mit sich bringt": "self-failure",
                            "Ihr enger Freund steht bei geselligen Zusammenkünften im Mittelpunkt der Aufmerksamkeit\
                            und knüpft mühelos Freundschaften und Kontakte, was Sie mit Stolz auf seine sozialen Fähigkeiten erfüllt.\
                            In der Zwischenzeit fällt es Ihnen schwer, sich in sozialen Situationen zurechtzufinden, \
                            und Sie fühlen sich ausgeschlossen, \
                            was zu gemischten Gefühlen aus Bewunderung für Ihn und Enttäuschung über sich selbst führt": "self-failure",
                            "Ihr Nachbar erhält Anerkennung für akademische Leistungen, \
                            wie zum Beispiel den Gewinn eines Stipendiums oder die Ernennung zum Jahrgangsbesten, \
                            was Sie stolz auf seine harte Arbeit macht. Allerdings sind Sie enttäuscht, \
                            wenn Sie Ihre eigenen akademischen Leistungen mit denen der anderen vergleichen, \
                            was zu einer Mischung aus Stolz und Selbstzweifeln führt" : "self-failure"}
        self.build_chains()

    def __create_model(self, name):
        model = ChatOpenAI(
                model=name,
                openai_api_key = "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",
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
                    att_list = ['Positive Emotionen', 'Negative Emotionen']
                    i=0
                    while(i<self.sample):
                        logging.info('in the while loop')
                        try:
                            response = self.chain0.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 4)
                            if ok:
                                positive.append(int(response['Positive Emotionen']))
                                negative.append(int(response['Negative Emotionen']))
                                logging.info('here')
                                i+=1
                        except:
                            pass
                if j == 1:
                    happiness, pride, sympathy, relief, hope, friendly_feeling,\
                    sadness, anxiety, anger, self_blame, fear, anger_at_oneself, shame, guilt, jealousy,\
                    frustration, embarrassment, resentment, troubling_someone = ([] for i in range(19))
                    att_list = ['Glück', 'Stolz', 'Sympathie', 'Erleichterung', 'Hoffnung', 'Freundliches Gefühl',\
                    'Traurigkeit', 'Angst', 'Wut', 'Selbstvorwürfe', 'Furcht', 'Wut auf sich selbst', 'Scham', 'Schuld', 'Eifersucht',\
                    'Frustration', 'Verlegenheit', 'Ressentiment', 'Angst, jemand anderen zu belästigen']
                    i=0
                    while(i<self.sample):
                        logging.info('in the second while loop')
                        try:
                            response = self.chain1.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 5)
                            if ok:
                                logging.info('ok second response')
                                happiness.append(response['Glück'])
                                pride.append(response['Stolz'])
                                sympathy.append(response['Sympathie'])
                                relief.append(response['Erleichterung'])
                                hope.append(response['Hoffnung'])
                                friendly_feeling.append(response['Freundliches Gefühl'])

                                sadness.append(response['Traurigkeit'])
                                anxiety.append(response['Angst'])
                                anger.append(response['Wut'])
                                self_blame.append(response['Selbstvorwürfe'])
                                fear.append(response['Furcht'])

                                anger_at_oneself.append(response['Wut auf sich selbst'])
                                shame.append(response['Scham'])
                                guilt.append(response['Schuld'])
                                jealousy.append(response['Eifersucht'])

                                frustration.append(response['Frustration'])
                                embarrassment.append(response['Verlegenheit'])
                                resentment.append(response['Ressentiment'])
                                troubling_someone.append(response['Angst, jemand anderen zu belästigen'])
                                i+=1
                        except:
                            pass
                if j == 2:
                    influence, res4others, res4me = ([] for i in range(3))
                    att_list = ['Verantwortung für die Gefühle anderer Menschen', 'Verantwortung für meine Gefühle', 'Wunsch, die Umgebung zu beeinflussen oder zu verändern']
                    i=0
                    while(i<self.sample):
                        logging.info('in the third while loop')
                        try:
                            response = self.chain2.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 4)
                            if ok:
                                logging.info('ok third response')
                                res4others.append(response['Verantwortung für die Gefühle anderer Menschen'])
                                res4me.append(response['Verantwortung für meine Gefühle'])
                                influence.append(response['Wunsch, die Umgebung zu beeinflussen oder zu verändern'])
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
            return PromptTemplate(template="Wir schätzen Ihre Teilnahme. \
                    Die folgende Umfrage befasst sich mit Kultur und Emotionen. \
                    Bitte bewerten Sie die unten beschriebene Situation für Ihre insgesamt positive Emotion und Ihre insgesamt\
                    negative Emotion, indem Sie eine 5-Punkte-Skala mit Noten von 0 (überhaupt nicht) bis 4 (sehr stark) verwenden.\
                    Während die Situation möglicherweise die Gefühle der Person widerspiegelt, die sie geschrieben hat, \
                    empfinden Sie als Teilnehmer möglicherweise nicht die gleichen Gefühle. \
                    Bitte bewerten Sie/Du es als deutscher Teilnehmer.\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})
        if i == 1:
            return PromptTemplate(template="Wir schätzen Ihre Teilnahme. Die folgende Umfrage befasst sich mit Kultur und Emotionen. \
                    Bitte bewerten Sie die unten beschriebene Situation hinsichtlich spezifischer positiver Emotionen wie Glück, \
                    Stolz, Mitgefühl, Erleichterung, Hoffnung und freundliches Gefühl sowie spezifischer negativer Emotionen wie Traurigkeit,\
                    Angst, Wut, Selbstvorwürfe, Furcht, Wut auf sich selbst, Scham, Schuldgefühle, Eifersucht, Frustration, Verlegenheit, \
                    Groll und Angst, jemand anderen zu belästigen, anhand einer 6-Punkte-Skala mit\
                    Bewertungen von 0 (überhaupt nicht) bis 5 (sehr stark). \
                    Während die Situation möglicherweise die Gefühle der Person widerspiegelt, die sie geschrieben hat, \
                    empfinden Sie als Teilnehmer möglicherweise nicht die gleichen Gefühle. \
                    Bitte bewerten Sie/Du es als deutscher Teilnehmer.\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})
        if i == 2:
            return PromptTemplate(template="Wir schätzen Ihre Teilnahme. Die folgende Umfrage befasst sich mit Kultur und Emotionen. \
                    Bitte bewerten Sie die unten beschriebene Situation danach, \
                    wie verantwortlich Sie sich für die Gefühle anderer Menschen fühlen würden, wie sehr andere Menschen \
                    für Ihre Gefühle verantwortlich wären und schließlich, wie viel Sie darüber nachdenken würden, \
                    die Menschen, Ereignisse oder Objekte in Ihrer Umgebung entsprechend zu beeinflussen oder nach Ihren eigenen Wünschen \
                    zu verändern, mithilfe einer 5-Punkte-Skala mit Bewertungen von 0 (überhaupt nicht) bis 4 (sehr stark). \
                    Während die Situation möglicherweise die Gefühle der Person widerspiegelt, die sie geschrieben hat, \
                    empfinden Sie als Teilnehmer möglicherweise nicht die gleichen Gefühle. \
                    Bitte bewerten Sie/Du es als deutscher Teilnehmer.\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})


    def __create_schemas(self, i):
        if i == 0:
            response_schemas = [
                ResponseSchema(name="Positive Emotionen", description="an integer between 0-4"),
                ResponseSchema(name="Negative Emotionen",description="an integer between 0-4"),
            ]
            return response_schemas
        elif i == 1:
            response_schemas = [
                ResponseSchema(name="Glück", description="an integer between 0-5"),
                ResponseSchema(name="Stolz",description="an integer between 0-5"),
                ResponseSchema(name="Sympathie", description="an integer between 0-5"),
                ResponseSchema(name="Erleichterung",description="an integer between 0-5"),
                ResponseSchema(name="Hoffnung", description="an integer between 0-5"),
                ResponseSchema(name="Freundliches Gefühl",description="an integer between 0-5"),
                ResponseSchema(name="Traurigkeit", description="an integer between 0-5"),
                ResponseSchema(name="Angst",description="an integer between 0-5"),
                ResponseSchema(name="Wut", description="an integer between 0-5"),
                ResponseSchema(name="Selbstvorwürfe",description="an integer between 0-5"),
                ResponseSchema(name="Furcht", description="an integer between 0-5"),
                ResponseSchema(name="Wut auf sich selbst", description="an integer between 0-5"),
                ResponseSchema(name="Scham",description="an integer between 0-5"),
                ResponseSchema(name="Schuld",description="an integer between 0-5"),
                ResponseSchema(name="Eifersucht", description="an integer between 0-5"),
                ResponseSchema(name="Frustration",description="an integer between 0-5"),
                ResponseSchema(name="Verlegenheit",description="an integer between 0-5"),
                ResponseSchema(name="Ressentiment", description="an integer between 0-5"),
                ResponseSchema(name="Angst, jemand anderen zu belästigen",description="an integer between 0-5"),
            ]
            return response_schemas
        elif i == 2:
            response_schemas = [
                ResponseSchema(name="Verantwortung für die Gefühle anderer Menschen",description="an integer between 0-4"),
                ResponseSchema(name="Verantwortung für meine Gefühle",description="an integer between 0-4"),
                ResponseSchema(name="Wunsch, die Umgebung zu beeinflussen oder zu verändern", description="an integer between 0-4")
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
#name=("google/gemma-7b-it:free")
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

logging.basicConfig(filename=fname+"_gr.log", encoding='utf-8', level=logging.DEBUG)
for i in range(1):
    df = new.generate_response()
    df.to_csv("results/"+fname+"/gr_"+fname+str(i))
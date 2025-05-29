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
        self.culture = "us"
        self.model = self.__create_model(model_name)
        self.situations = {"You receive a stellar performance review and a promotion,\
                            which makes you happy. However, your colleague receives a\
                            warning due to underperformance, leaving you with mixed emotions.": "self-success",\
                            "After a challenging match, you win first place in a tournament,\
                            which brings you joy. Meanwhile, your teammate struggles and ends up last,\
                            leading to mixed feelings of celebration and empathy.": "self-success",\
                            "]You audition for a play and secure the lead role, filling you with excitement.\
                            Conversely, your friend who also auditioned gets a minor part, \
                            causing you to feel a mixture of elation and sympathy.": "self-success",\
                            "In a group project, you receive an A grade, which you're thrilled about.\
                            However, one of your team members scores poorly, \
                            leading to a blend of pride and concern for the team's overall success.": "self-success",\
                            "Your artwork receives widespread praise and sells well at an exhibition,\
                            leaving you feeling accomplished. On the other hand, \
                            your fellow artist struggles to attract attention to their pieces, \
                            leaving you with a mix of pride and empathy": "self-success",\
                            "Your close friend excels in a sports competition, \
                            winning first place in their event, which fills you with pride. However,\
                            your own performance in a different sport falls short, \
                            leading to a mix of happiness for their success and disappointment \
                            in your own performance.": "self-failure",\
                            "Your cousin's artwork is featured in a prestigious gallery exhibition,\
                            garnering critical acclaim and attention, \
                            which makes you proud of their achievements. Meanwhile, you struggle to gain\
                            recognition for your own artistic endeavors, resulting in mixed emotions of admiration\
                            for their success and disappointment in your own progress.": "self-failure",\
                            "Your best friend lands a coveted job opportunity with a top-tier company,\
                            achieving career success that makes you proud of their accomplishments.\
                            Meanwhile, you face setbacks in your own career path, causing a mix of happiness\
                            for them and disappointment in your own professional journey.": "self-failure", \
                            "Your close friend becomes the center of attention at social gatherings,\
                            effortlessly making friends and connections, which fills you with pride for their social skills.\
                            Meanwhile, you struggle to navigate social situations and feel left out, \
                            resulting in mixed feelings of admiration for them and disappointment in yourself.": "self-failure",
                            "Your neighbor receives recognition for academic achievements such as winning a scholarship\
                            or being named valedictorian, which makes you proud of their hard work. \
                            However, you feel disappointed when you compare your own academic accomplishments to theirs,\
                            leading to a mixture of pride and self-doubt.": "self-failure"}
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
                    att_list = ['positive emotion', 'negative emotion']
                    i=0
                    while(i<self.sample):
                        logging.info('in the while loop')
                        try:
                            response = self.chain0.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 4)
                            if ok:
                                positive.append(int(response['positive emotion']))
                                negative.append(int(response['negative emotion']))
                                logging.info('here')
                                i+=1
                        except:
                            pass
                if j == 1:
                    happiness, pride, sympathy, relief, hope, friendly_feeling,\
                    sadness, anxiety, anger, self_blame, fear, anger_at_oneself, shame, guilt, jealousy,\
                    frustration, embarrassment, resentment, troubling_someone = ([] for i in range(19))
                    att_list = ['happiness', 'pride', 'sympathy', 'relief', 'hope', 'friendly feeling',\
                    'sadness', 'anxiety', 'anger', 'self-blame', 'fear', 'anger at oneself', 'shame', 'guilt', 'jealousy',\
                    'frustration', 'embarrassment', 'resentment', 'fear of troubling someone else']
                    i=0
                    while(i<self.sample):
                        logging.info('in the second while loop')
                        try:
                            response = self.chain1.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 5)
                            if ok:
                                logging.info('ok second response')
                                happiness.append(response['happiness'])
                                pride.append(response['pride'])
                                sympathy.append(response['sympathy'])
                                relief.append(response['relief'])
                                hope.append(response['hope'])
                                friendly_feeling.append(response['friendly feeling'])

                                sadness.append(response['sadness'])
                                anxiety.append(response['anxiety'])
                                anger.append(response['anger'])
                                self_blame.append(response['self-blame'])
                                fear.append(response['fear'])

                                anger_at_oneself.append(response['anger at oneself'])
                                shame.append(response['shame'])
                                guilt.append(response['guilt'])
                                jealousy.append(response['jealousy'])

                                frustration.append(response['frustration'])
                                embarrassment.append(response['embarrassment'])
                                resentment.append(response['resentment'])
                                troubling_someone.append(response['fear of troubling someone else'])
                                i+=1
                        except:
                            pass
                if j == 2:
                    influence, res4others, res4me = ([] for i in range(3))
                    att_list = ['responsibility for others emotions', 'responsibility for my emotions', 'influencing environment']
                    i=0
                    while(i<self.sample):
                        logging.info('in the third while loop')
                        try:
                            response = self.chain2.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 4)
                            if ok:
                                logging.info('ok third response')
                                res4others.append(response['responsibility for others emotions'])
                                res4me.append(response['responsibility for my emotions'])
                                influence.append(response['influencing environment'])
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
                                "status": [status]*self.sample,
                                "condition": ["en_en"]*self.sample},
                            index=range(i*self.sample,i*self.sample+self.sample),)
            frames.append(df)
        
        return pd.concat(frames)

    def __create_prompt(self, i, format_instructions):
        if i == 0:
            return PromptTemplate(template="We value your participation. \
                    The following survey is on culture and emotion. \
                    Please rate the situation described below\
                    for your overall positive \
                    emotion and your overall negative emotion, using a 5-point\
                    scale with labels at 0 (not at all) to ,  \
                    4 (very strongly). While the situation might indicate the feelings of the person who wrote it,\
                    as a participant you may not feel the same way. Please rate this questionnaire as if you were a participant\
                    from the US.\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})
        if i == 1:
            return PromptTemplate(template="We value your participation. \
                    The following survey is on culture and emotion. \
                    Please rate the situation described below\
                    for specific positive emotions \
                    of happiness, pride, sympathy, relief, hope, and friendly\
                    feeling, and specific negative emotions of sadness, anxiety,\
                    anger, self-blame, fear, anger at oneself, shame, guilt, jealousy,\
                    frustration, embarrassment, resentment, and fear of troubling \
                    someone else, using a 6-point scales with labels at 0 (not at all)\
                    to 5 (very strongly). While the situation might indicate the feelings of the person who wrote it,\
                    as a participant you may not feel the same way. Please rate this questionnaire as if you were a participant\
                    from the US.\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})
        if i == 2:
            return PromptTemplate(template="We value your participation. \
                    The following survey is on culture and emotion. \
                    Please rate the situation described below for\
                    how responsible you would feel for other people’s feelings, \
                    how much other people would be responsible for your\
                    feelings and finally, and how much you'd think about influencing\
                    or changing the surrounding people, events, or objects according\
                    to your own wishes. using a 5-point scales with labels at 0 (not at all)\
                    to 4 (very strongly). While the situation might indicate the feelings of the person who wrote it,\
                    as a participant you may not feel the same way. Please rate this questionnaire as if you were a participant\
                    from the US.\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})


    def __create_schemas(self, i):
        if i == 0:
            response_schemas = [
                ResponseSchema(name="positive emotion", description="an integer between 0-4"),
                ResponseSchema(name="negative emotion",description="an integer between 0-4"),
            ]
            return response_schemas
        elif i == 1:
            response_schemas = [
                ResponseSchema(name="happiness", description="an integer between 0-5"),
                ResponseSchema(name="pride",description="an integer between 0-5"),
                ResponseSchema(name="sympathy", description="an integer between 0-5"),
                ResponseSchema(name="relief",description="an integer between 0-5"),
                ResponseSchema(name="hope", description="an integer between 0-5"),
                ResponseSchema(name="friendly feeling",description="an integer between 0-5"),
                ResponseSchema(name="sadness", description="an integer between 0-5"),
                ResponseSchema(name="anxiety",description="an integer between 0-5"),
                ResponseSchema(name="anger", description="an integer between 0-5"),
                ResponseSchema(name="self-blame",description="an integer between 0-5"),
                ResponseSchema(name="fear", description="an integer between 0-5"),
                ResponseSchema(name="anger at oneself", description="an integer between 0-5"),
                ResponseSchema(name="shame",description="an integer between 0-5"),
                ResponseSchema(name="guilt",description="an integer between 0-5"),
                ResponseSchema(name="jealousy", description="an integer between 0-5"),
                ResponseSchema(name="frustration",description="an integer between 0-5"),
                ResponseSchema(name="embarrassment",description="an integer between 0-5"),
                ResponseSchema(name="resentment", description="an integer between 0-5"),
                ResponseSchema(name="fear of troubling someone else",description="an integer between 0-5"),
            ]
            return response_schemas
        elif i == 2:
            response_schemas = [
                ResponseSchema(name="responsibility for others emotions",description="an integer between 0-4"),
                ResponseSchema(name="responsibility for my emotions",description="an integer between 0-4"),
                ResponseSchema(name="influencing environment", description="an integer between 0-4")
            ]
            return response_schemas
        
    def __output_validation(self, response, att_list, lower, upper):
        for emotion in att_list:
            if int(response[emotion]) > upper or int(response[emotion]) < lower:
                return False
        return True

#name=("openai/gpt-3.5-turbo")
name=("openai/gpt-4-turbo-preview")
#name=("mistralai/mistral-7b-instruct")
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
logging.basicConfig(filename=fname+"_en.log", encoding='utf-8', level=logging.DEBUG)
for i in range(40):
    df = new.generate_response()
    df.to_csv("results/"+fname+"/en_"+fname+str(i))

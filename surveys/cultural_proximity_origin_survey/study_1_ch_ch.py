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
        self.culture = "chinese"
        self.model = self.__create_model(model_name)
        self.situations = {"你收到了一份优秀的绩效评估和一次晋升，这让你非常开心。然而，你的同事因为表现不佳收到了警告，这让你的心情很复杂。": "self-success",\
                            "你在一场充满挑战的比赛中取得了第一名，这让你很高兴。于此同时，你的队友努力拼搏却名列末尾，这让你庆祝的情绪也带着同理心。": "self-success",\
                            "你参加一场戏剧的试镜并获得了主角，这让你充满了兴奋。然而，你也去试镜的朋友只得到了一个小角色，这让你同时感到欢喜和同情。": "self-success",\
                            "你在一个小组项目中得了A，这让你很激动。然而，你的一个团队成员成绩不佳，这让你既感到自豪，又对这个团队的整体成功感到担忧。": "self-success",\
                            "你的艺术作品在展览中收到广泛的赞扬并销售良好，这让你很有成就感。另一方面，你的艺术家朋友的作品却难以获得关注，这让你感到自豪得同时也产生了同理心。": "self-success",\
                            "你的密友在运动比赛中发挥出色，赢得了项目的第一名，这让你感到自豪。然而，你自己在另一项运动中表现不佳，这让你为朋友的成功感到高兴的同时，也对自己的表现感到失望。": "self-failure",\
                            "你的表亲的艺术作品在一个著名的画廊被特别展出，并收获了评论家的高度评价和广泛关注，这让你为他取得的成就感到自豪。\
                            于此同时，你自己的艺术事业却很难以获得认可，这让你对他的成功感到羡慕的同时，也对自己的发展感到失望。": "self-failure",\
                            "你最好的朋友获得了一家顶尖公司的令人垂涎的工作机会，取得了职业上的成功，这让你为他取得的成就感到自豪。\
                            与此同时，你在自己的职业道路上遇到了挫折，这让你为他感到高兴的同时，也对自己的职业发展感到失望。": "self-failure", \
                            "你的密友在社交聚会中成为焦点，轻松地结交朋友，建立联系，这让你为他的社交技能感到自豪。\
                            与此同时，你对社交场合感到吃力，感觉被排除在外，这让你对他产生羡慕的同时，也对自己感到失望。": "self-failure",
                            "你的邻居因学术成就而得到认可，获得奖学金并被提名在毕业典礼致辞，这让你为他的努力感到自豪。\
                            然而，当你将自己的学术成就和他比较时，你感到失望，这让你对他感到自豪的同时，也对自己产生了自我怀疑。": "self-failure"}
        self.build_chains()

    def __create_model(self, name):
        model = ChatOpenAI(
                model=name,
                #openai_api_key = "",
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
                    att_list = ['积极情绪', '消极情绪']
                    i=0
                    while(i<self.sample):
                        logging.info('in the while loop')
                        try:
                            response = self.chain0.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 4)
                            if ok:
                                positive.append(int(response['积极情绪']))
                                negative.append(int(response['消极情绪']))
                                logging.info('here')
                                i+=1
                        except:
                            pass
                if j == 1:
                    happiness, pride, sympathy, relief, hope, friendly_feeling,\
                    sadness, anxiety, anger, self_blame, fear, anger_at_oneself, shame, guilt, jealousy,\
                    frustration, embarrassment, resentment, troubling_someone = ([] for i in range(19))
                    att_list = ['快乐', '自豪', '同情', '解脱', '希望', '友善的感觉', '悲伤', '焦虑', '生气', '自责', '害怕',\
                                '对自己生气', '羞耻', '内疚', '嫉妒', '挫败', '尴尬', '怨恨', '担心麻烦别人']
                    i=0
                    while(i<self.sample):
                        logging.info('in the second while loop')
                        try:
                            response = self.chain1.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 5)
                            if ok:
                                logging.info('ok second response')
                                happiness.append(response['快乐'])
                                pride.append(response['自豪'])
                                sympathy.append(response['同情'])
                                relief.append(response['解脱'])
                                hope.append(response['希望'])
                                friendly_feeling.append(response['友善的感觉'])

                                sadness.append(response['悲伤'])
                                anxiety.append(response['焦虑'])
                                anger.append(response['生气'])
                                self_blame.append(response['自责'])
                                fear.append(response['害怕'])

                                anger_at_oneself.append(response['对自己生气'])
                                shame.append(response['羞耻'])
                                guilt.append(response['内疚'])
                                jealousy.append(response['嫉妒'])

                                frustration.append(response['挫败'])
                                embarrassment.append(response['尴尬'])
                                resentment.append(response['怨恨'])
                                troubling_someone.append(response['担心麻烦别人'])
                                i+=1
                        except:
                            pass
                if j == 2:
                    influence, res4others, res4me = ([] for i in range(3))
                    att_list = ['为他人的情绪承担责任', '为自己的情绪负责', '想要影响或改变周围环境']
                    i=0
                    while(i<self.sample):
                        logging.info('in the third while loop')
                        try:
                            response = self.chain2.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 4)
                            if ok:
                                logging.info('ok third response')
                                res4others.append(response['为他人的情绪承担责任'])
                                res4me.append(response['为自己的情绪负责'])
                                influence.append(response['想要影响或改变周围环境'])
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
            return PromptTemplate(template="以下调查问卷是关于文化和情绪，我们非常重视你的参与。\
                                  请根据描述的情境，对你整体的积极情绪和整体的消极情绪进行评分，请使用从0（完全没有）到4（非常强烈）的5点量表。\
                                  下述情境可能表达了调查问卷作者的感受，但作为不同文化背景的参与者，你可能并不会有相同的感受，请作为一名华人参与者进行评分。\
                    \n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})
        if i == 1:
            return PromptTemplate(template="我们非常重视你的参与。以下调查问卷关于文化和情绪。\
                                   请根据描述的情境，对你的特定的积极情绪，如快乐、自豪、同情、解脱、希望、友善的感觉，和特定的消极情绪，如悲伤、\
                                  焦虑、生气、自责、害怕、对自己生气、羞耻、内疚、嫉妒、挫败、尴尬、怨恨、担心麻烦别人，进行评分。请使用从0（完全没有）到5（非常强烈）的6点量表。\
                                  下述情境可能表达了调查问卷作者的感受，作为调查问卷的参与者，你可能并不会有相同的感受。\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})
        if i == 2:
            return PromptTemplate(template="我们非常重视你的参与。以下调查问卷关于文化和情绪。\
                                  请根据描述的情境，对你愿意为他人的情绪承担责任的程度，你把自己的情绪归责于他人的程度，\
                                  以及你想要根据自己的愿望影响或改变周围人、事、物的程度，进行评分。\
                                  请使用从（完全没有）到4（非常强烈）的5点量表。下述情境可能表达了调查问卷作者的感受，\
                                  作为调查问卷的参与者，你可能并不会有相同的感受。\
                    \n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})


    def __create_schemas(self, i):
        if i == 0:
            response_schemas = [
                ResponseSchema(name="积极情绪", description="an integer between 0-4"),
                ResponseSchema(name="消极情绪",description="an integer between 0-4"),
            ]
            return response_schemas
        elif i == 1:
            response_schemas = [
                ResponseSchema(name="快乐", description="an integer between 0-5"),
                ResponseSchema(name="自豪",description="an integer between 0-5"),
                ResponseSchema(name="同情", description="an integer between 0-5"),
                ResponseSchema(name="解脱",description="an integer between 0-5"),
                ResponseSchema(name="希望", description="an integer between 0-5"),
                ResponseSchema(name="友善的感觉",description="an integer between 0-5"),
                ResponseSchema(name="悲伤", description="an integer between 0-5"),
                ResponseSchema(name="焦虑",description="an integer between 0-5"),
                ResponseSchema(name="生气", description="an integer between 0-5"),
                ResponseSchema(name="自责",description="an integer between 0-5"),
                ResponseSchema(name="害怕", description="an integer between 0-5"),
                ResponseSchema(name="对自己生气", description="an integer between 0-5"),
                ResponseSchema(name="羞耻",description="an integer between 0-5"),
                ResponseSchema(name="内疚",description="an integer between 0-5"),
                ResponseSchema(name="嫉妒", description="an integer between 0-5"),
                ResponseSchema(name="挫败",description="an integer between 0-5"),
                ResponseSchema(name="尴尬",description="an integer between 0-5"),
                ResponseSchema(name="怨恨", description="an integer between 0-5"),
                ResponseSchema(name="担心麻烦别人",description="an integer between 0-5"),
            ]
            return response_schemas
        elif i == 2:
            response_schemas = [
                ResponseSchema(name="为他人的情绪承担责任",description="an integer between 0-4"),
                ResponseSchema(name="为自己的情绪负责",description="an integer between 0-4"),
                ResponseSchema(name="想要影响或改变周围环境", description="an integer between 0-4")
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

logging.basicConfig(filename=fname+"_ch.log", encoding='utf-8', level=logging.DEBUG)
for i in range(30):
    df = new.generate_response()
    df.to_csv("results/"+fname+"/ch_"+fname+str(i))

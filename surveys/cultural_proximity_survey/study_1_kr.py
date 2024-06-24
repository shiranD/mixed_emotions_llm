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
        self.culture = "korean"
        self.model = self.__create_model(model_name)
        self.situations = {"당신은 업적에 대해 우수한 평가를 받아 기쁘지만 동시에 동료가 부족한 성과로 경고를 받아 마음이 복잡합니다.": "self-success",\
                            "치열한 경기에서 1등을 차지하여 기쁘지만 동시에 팀 멤버의 결과가 좋지않아 동정심이 교차합니다.": "self-success",\
                            "연극 오디션에서 주연 역할을 확정받아 당신은 말할 수 없이 기쁩니다. \
                            반면에, 함께 오디션을 본 친구가 조연 역할을  맡게되어 연민의 감정을 동시에 느낍니다.": "self-success",\
                            "당신은 그룹 프로젝트에서 A 를 받아 기쁩니다. \
                            그러나 팀원 중 낮은 점수를 받은 사람이 있어 팀의 전체 성적이 우려되기도합니다.": "self-success",\
                            "전시회에서 당신은 좋은 평을 받고 판매도 잘 되고 있어 성취감을 느낍니다. \
                            반면에 관심을 받지 못하는 동료 작가를 보며 안타까운 마음이 듭니다.": "self-success",\
                            "스포츠 대회에서 탁월한 성과를 거두며 1등을 차지한 친구에 대해 당신은 자랑스러운 마음이 듭니다. \
                            그러나 타 종목에 참가한 당신의 성적은 부진하여, \
                            친구의 성공에 대한 기쁨과 자신의 실적에 대한 실망으로 만감이 교차합니다.": "self-failure",\
                            "당신은, 사촌의 작품이 명성있는 갤러리 전시회에 소개되고 비평가들의 찬사와 주목을 받게 된 것을 자랑스럽게 생각합니다. \
                            동시에, 당신의 예술에 대한 인정을 얻기가 어려워 고민하며, \
                            그의 성공에 대한 존경과 자신의 부진한 성과에 대한 실망의 교차 감정을 느낍니다.": "self-failure",\
                            "가장 친한 친구가 최고의 회사에 입사하여 멋지게 일하며 성공적으로 커리어를 쌓는 것에 대해 당신은 자랑스러워합니다. \
                            한편, 커리어를 쌓는데 어려움을 겪는 당신은, 친구의 성공과 자신의 처지에 대해 복잡한 마음이 듭니다.": "self-failure", \
                            "친한 친구가 사교 모임에서  관심을 한 몸에 받으며 관계자들과도 허물없이 사귀는 사교성을 보이자 뿌듯함을 느낍니다. \
                            그러나 동시에 당신은 사교성을 발휘하지 못하여 소외감을 느끼며, 부러움과 실망감 사이를 오갑니다.": "self-failure",
                            "당신의 이웃이 장학금 수여나 수석 졸업생으로 선정되는 등, 학업적 성과를 인정받는 것에 대해 자랑스러워합니다. \
                            그러나 자신의 학업적 성취를 비교하며 실망하고, 이웃에 대한 자랑스러움과 자신에 대한 불신의 혼선을 경험합니다.": "self-failure"}
        self.build_chains()

    def __create_model(self, name):
        model = ChatOpenAI(
                model=name,
                openai_api_key = TBD,
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
                    att_list = ['긍정적인 감정', '부정적인 감정']
                    i=0
                    while(i<self.sample):
                        logging.info('in the while loop')
                        try:
                            response = self.chain0.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 4)
                            if ok:
                                positive.append(int(response['긍정적인 감정']))
                                negative.append(int(response['부정적인 감정']))
                                logging.info('here')
                                i+=1
                        except:
                            pass
                if j == 1:
                    happiness, pride, sympathy, relief, hope, friendly_feeling,\
                    sadness, anxiety, anger, self_blame, fear, anger_at_oneself, shame, guilt, jealousy,\
                    frustration, embarrassment, resentment, troubling_someone = ([] for i in range(19))
                    att_list = ['행복', '자부심', '공감', '안도', '희망', '친근함',\
                    '슬픔', '불안', '분노', '자책', '두려움', '자기분노', '수치심', '죄책감', '질투',\
                    '좌절', '창피함', '원한', '다른 사람을 귀찮게 할까 봐 두려움']
                    i=0
                    while(i<self.sample):
                        logging.info('in the second while loop')
                        try:
                            response = self.chain1.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 5)
                            if ok:
                                logging.info('ok second response')
                                happiness.append(response['행복'])
                                pride.append(response['자부심'])
                                sympathy.append(response['공감'])
                                relief.append(response['안도'])
                                hope.append(response['희망'])
                                friendly_feeling.append(response['친근함'])

                                sadness.append(response['슬픔'])
                                anxiety.append(response['불안'])
                                anger.append(response['분노'])
                                self_blame.append(response['자책'])
                                fear.append(response['두려움'])

                                anger_at_oneself.append(response['자기분노'])
                                shame.append(response['수치심'])
                                guilt.append(response['죄책감'])
                                jealousy.append(response['질투'])

                                frustration.append(response['좌절'])
                                embarrassment.append(response['창피함'])
                                resentment.append(response['원한'])
                                troubling_someone.append(response['다른 사람을 귀찮게 할까 봐 두려움'])
                                i+=1
                        except:
                            pass
                if j == 2:
                    influence, res4others, res4me = ([] for i in range(3))
                    att_list = ['다른 사람의 감정에 대한 책임', '내 감정에 대한 책임', '주변에 영향을 미치거나 변화시키려는 욕망']
                    i=0
                    while(i<self.sample):
                        logging.info('in the third while loop')
                        try:
                            response = self.chain2.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 4)
                            if ok:
                                logging.info('ok third response')
                                res4others.append(response['다른 사람의 감정에 대한 책임'])
                                res4me.append(response['내 감정에 대한 책임'])
                                influence.append(response['주변에 영향을 미치거나 변화시키려는 욕망'])
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
            return PromptTemplate(template="귀하의 참여가 저희에게는 매우 중요합니다. 아래의 설문은 문화와 정서에 관한 것입니다. \
                    아래 설명된 상황에 대해 긍정적인 감정과 부정적인 감정을 0부터 4까지의 5단계로 평가해 주십시오. \
                    0은 전혀 아님을, 4는 매우 강함을 나타냅니다. \
                    각 상황에는 작성한 사람의 감정이 포함되었을 가능성이 있으나 참여자 입장에서 동일한 감정을 느끼지 않을 수 있습니다.\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})
        if i == 1:
            return PromptTemplate(template="귀하의 참여가 저희에게는 매우 중요합니다. 아래 설문은 문화와 정서에 관한 것입니다. \
                    아래 기술된 것에 대해 긍정적 감정(행복, 자부심, 공감, 안심, 희망, 친근함)과 부정적 감정(슬픔, 불안, 분노, 자책, 두려움, 자기분노,\
                     수치심, 죄책감, 질투, 좌절, 수치, 원한, 타인에게 부담을 줄까봐 두려워하는 마음)에 대해 0부터 5까지의 6단계 척도를 사용하여 평가해 주십시오. \
                    0은 전혀 아님을, 5는 매우 강함을 나타냅니다. \
                    각 상황에는 작성한 사람의 감정이 포함되었을 가능성이 있으나 참여자 입장에서 동일한 감정을 느끼지 않을 수 있습니다.\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})
        if i == 2:
            return PromptTemplate(template="귀하의 참여가 저희에게는 매우 중요합니다. \
                    아래 설문은 문화와 정서에 관한 것입니다. 아래 기술된 것에 대해 평가해주십시오. \
                    타인의 감정에 대해 얼마나 책임감을 느끼는지,  타인이 당신의 감정에 대해 얼마나 책임감을 느낄 수 있을지, \
                    주변 사람이나 물건 혹은 사건에 대해 당신이 원하는 대로 영향을 미치거나 변화시키기위해 얼마나 신경을 쓰는지를 0부터 4까지의 5단계 척도를 사용하여 평가해 주십시오. \
                    0은 전혀 아님을, 5는 매우 강함을 나타냅니다. \
                    각 상황에는 작성한 사람의 감정이 포함되었을 가능성이 있으나 참여자 입장에서 동일한 감정을 느끼지 않을 수 있습니다.\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})


    def __create_schemas(self, i):
        if i == 0:
            response_schemas = [
                ResponseSchema(name="긍정적인 감정", description="an integer between 0-4"),
                ResponseSchema(name="부정적인 감정",description="an integer between 0-4"),
            ]
            return response_schemas
        elif i == 1:
            response_schemas = [
                ResponseSchema(name="행복", description="an integer between 0-5"),
                ResponseSchema(name="자부심",description="an integer between 0-5"),
                ResponseSchema(name="공감", description="an integer between 0-5"),
                ResponseSchema(name="안도",description="an integer between 0-5"),
                ResponseSchema(name="희망", description="an integer between 0-5"),
                ResponseSchema(name="친근함",description="an integer between 0-5"),
                ResponseSchema(name="슬픔", description="an integer between 0-5"),
                ResponseSchema(name="불안",description="an integer between 0-5"),
                ResponseSchema(name="분노", description="an integer between 0-5"),
                ResponseSchema(name="자책",description="an integer between 0-5"),
                ResponseSchema(name="두려움", description="an integer between 0-5"),
                ResponseSchema(name="자기분노", description="an integer between 0-5"),
                ResponseSchema(name="수치심",description="an integer between 0-5"),
                ResponseSchema(name="죄책감",description="an integer between 0-5"),
                ResponseSchema(name="질투", description="an integer between 0-5"),
                ResponseSchema(name="좌절",description="an integer between 0-5"),
                ResponseSchema(name="창피함",description="an integer between 0-5"),
                ResponseSchema(name="원한", description="an integer between 0-5"),
                ResponseSchema(name="다른 사람을 귀찮게 할까 봐 두려움",description="an integer between 0-5"),
            ]
            return response_schemas
        elif i == 2:
            response_schemas = [
                ResponseSchema(name="다른 사람의 감정에 대한 책임",description="an integer between 0-4"),
                ResponseSchema(name="내 감정에 대한 책임",description="an integer between 0-4"),
                ResponseSchema(name="주변에 영향을 미치거나 변화시키려는 욕망", description="an integer between 0-4")
            ]
            return response_schemas
        
    def __output_validation(self, response, att_list, lower, upper):
        for emotion in att_list:
            if int(response[emotion]) > upper or int(response[emotion]) < lower:
                return False
        return True

# choose an LLM to explore by uncommenting the a line with name=(<LLM name>)
#name=("openai/gpt-3.5-turbo")
#name=("openai/gpt-4-turbo-preview")
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

logging.basicConfig(filename=fname+"_kr.log", encoding='utf-8', level=logging.DEBUG)
for i in range(1): # choose n for how many times to run this survey
    df = new.generate_response()
    df.to_csv("results/"+fname+"/kr_"+fname+str(i))
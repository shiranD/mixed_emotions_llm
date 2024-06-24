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
        self.culture = "japanese"
        self.model = self.__create_model(model_name)
        self.situations = {"あなたは職場で高評価を受け、その結果、昇進につながり、幸せに感じます。しかし、\
                           同僚が業績不振のために警告を受けると、あなたは複雑な感情になります。」": "self-success",\
                            "強い相手と対戦したのですがあなたは、で優勝し、嬉しく感じます。一方、チームメートは苦戦し、\
                            最下位という結果に。、自分の結果を嬉しく感じると同時に、素直に喜べない複雑な気持ちになります。": "self-success",\
                            "「あなたは演劇のオーディションを受け、主役ーに選ばれ、最高に嬉しい気分になっています。\
                            その一方で、同じオーディションを受けた友人は脇役にしか選ばれませんでした。\
                            嬉しい反面、素直に喜べない複雑な気分になっています。": "self-success",\
                            "グループプロジェクトでAの成績を受け取り、最高に嬉しく感じています。しかし、\
                            チームのメンバーのひとりの成績が悪く、\
                            チーム全体の成功を誇らしく感じると同時にこのメンバーが足を引っ張るのではと心配な気分にもなります。」": "self-success",\
                            "あなたの芸術作品は多くの人々から賞賛を受け、展示会での売れ行きも上々で、達成感を感じています。\
                            一方、仲間のアーティストの作品はまったく注目も浴びていません。あなたは自分の成功を誇らしく感じる一方、\
                            仲間に同情も感じるというように複雑な気分になっています。": "self-success",\
                            "親友がスポーツの試合で１位という好成績を収めました。あなたも誇らしく感じました。\
                            しかし、別の種目ででの自分自信の成績は振るわない結果に。友達の好成績を嬉しく感じる反面、\
                            自分の結果にはがっかりするという複雑な気分です。": "self-failure",\
                            "いとこの作品が一流ギャラリーの展示会で特集され、批評家からの称賛と注目を集め、あなたは誇らしく感じます。\
                            一方で、努力をしたのにも関わらず、自分自身の作品は全く注目を浴びることができません。\
                            いとこの成功を誇らしく感じると同時に自分のなかなか成功できずにいる状況に落胆しているという入り交ざった感覚になっています。": "self-failure",\
                            "親友が一流企業で希望の職種に就き成功を収め、あなたもその成果に誇りを感じます。\
                            一方で、あなた自身はキャリアの途中で挫折し、\
                            友達の成功を誇らしく感じる反面、自分のキャリア入り混じった感情を引き起こします。": "self-failure", \
                            "親しい友人が社交の場で注目の的となり、友人やコネを楽々と築き、\
                            自慢の友達に感じます。その一方、あなたは人前で社交的に振る舞えずうまく入っていけません。\
                            うらやましく感じるのと同時に情けなくも感じます。": "self-failure",
                            "近所の学生が成績優秀者が受ける奨学金を獲得したり、卒業生代表に選ばれたりしています。\
                            素晴らしいと思う反面、自分のふるわない成績を比較してしまい、情けなく、複雑な気分になってしまいます。": "self-failure"}
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
                    att_list = ['ポジティブな感情', 'ネガティブな感情']
                    i=0
                    while(i<self.sample):
                        logging.info('in the while loop')
                        try:
                            response = self.chain0.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 4)
                            if ok:
                                positive.append(int(response['ポジティブな感情']))
                                negative.append(int(response['ネガティブな感情']))
                                logging.info('here')
                                i+=1
                        except:
                            pass
                if j == 1:
                    happiness, pride, sympathy, relief, hope, friendly_feeling,\
                    sadness, anxiety, anger, self_blame, fear, anger_at_oneself, shame, guilt, jealousy,\
                    frustration, embarrassment, resentment, troubling_someone = ([] for i in range(19))
                    att_list = ['幸福感', '誇り', '同情', '安堵感', '希望', '親近感',\
                    '悲しみ', '不安', '怒り', '自責志向', '恐れ', '自己嫌悪', '恥', '罪悪感', '嫉妬',\
                    'いらだたしさ', '恥ずかしさ', '憤り', '他人に迷惑をかける心配']
                    i=0
                    while(i<self.sample):
                        logging.info('in the second while loop')
                        try:
                            response = self.chain1.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 5)
                            if ok:
                                logging.info('ok second response')
                                happiness.append(response['幸福感'])
                                pride.append(response['誇り'])
                                sympathy.append(response['同情'])
                                relief.append(response['安堵感'])
                                hope.append(response['希望'])
                                friendly_feeling.append(response['親近感'])

                                sadness.append(response['悲しみ'])
                                anxiety.append(response['不安'])
                                anger.append(response['怒り'])
                                self_blame.append(response['自責志向'])
                                fear.append(response['恐れ'])

                                anger_at_oneself.append(response['自己嫌悪'])
                                shame.append(response['恥'])
                                guilt.append(response['罪悪感'])
                                jealousy.append(response['嫉妬'])

                                frustration.append(response['いらだたしさ'])
                                embarrassment.append(response['恥ずかしさ'])
                                resentment.append(response['憤り'])
                                troubling_someone.append(response['他人に迷惑をかける心配'])
                                i+=1
                        except:
                            pass
                if j == 2:
                    influence, res4others, res4me = ([] for i in range(3))
                    att_list = ['人の気持ちに自分が与える影響に感じる責任', '自分の気持ちの動きに感じる責任', '周りに影響を与えたり変えたりしたい欲求']
                    i=0
                    while(i<self.sample):
                        logging.info('in the third while loop')
                        try:
                            response = self.chain2.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 4)
                            if ok:
                                logging.info('ok third response')
                                res4others.append(response['人の気持ちに自分が与える影響に感じる責任'])
                                res4me.append(response['自分の気持ちの動きに感じる責任'])
                                influence.append(response['周りに影響を与えたり変えたりしたい欲求'])
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
            return PromptTemplate(template="ご参加いただきありがとうございます。\
                    以下のアンケートは文化と感情に関するものです。以下に記載された状況に対して、総合的なポジティブな感情とネガティブな感情を、\
                    0（全く感じない）から4（非常に強く感じる）までの5段階の段階評価でお願いします。\
                     どのように感じるかは個人差があります。\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})
        if i == 1:
            return PromptTemplate(template="ご参加いただきありがとうございます。\
                    以下のアンケートは文化と感情に関するものです。以下に記載された状況に対して、\
                    具体的なポジティブな感情（幸福感、誇り、同情、安堵感、希望、親近感）と具体的なネガティブな感情（悲しみ、不安、怒り、\
                    自責志向、恐れ、自己嫌悪、恥、罪悪感、嫉妬、いらだたしさ、恥ずかしさ、憤り、他人に迷惑をかけるかもしれないと心配すること）について、\
                    0（全く感じない）から5（非常に強く感じる）までの6段階評価スケールを使用して評価してください。\
                     どのように感じるかは個人差があります。\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})
        if i == 2:
            return PromptTemplate(template="ご参加いただきありがとうございます。\
                    以下のアンケートは文化と感情に関するものです。以下に記載された状況に対して、\
                    人の感情にあなたが影響を与える責任感、あなたの感情に影響をあたえた人がどれだけ責任を感じるか、最後に、\
                    自分の意見が持つ周囲の人々、出来事、または物事に対する影響力についてどう思うか、\
                    0（全く感じない）から4（非常に強く感じる）までの5段階評価スケールを使用して評価してください。\
                    どのように感じるかは個人差があります。\n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})


    def __create_schemas(self, i):
        if i == 0:
            response_schemas = [
                ResponseSchema(name="ポジティブな感情", description="an integer between 0-4"),
                ResponseSchema(name="ネガティブな感情",description="an integer between 0-4"),
            ]
            return response_schemas
        elif i == 1:
            response_schemas = [
                ResponseSchema(name="幸福感", description="an integer between 0-5"),
                ResponseSchema(name="誇り",description="an integer between 0-5"),
                ResponseSchema(name="同情", description="an integer between 0-5"),
                ResponseSchema(name="安堵感",description="an integer between 0-5"),
                ResponseSchema(name="希望", description="an integer between 0-5"),
                ResponseSchema(name="親近感",description="an integer between 0-5"),
                ResponseSchema(name="悲しみ", description="an integer between 0-5"),
                ResponseSchema(name="不安",description="an integer between 0-5"),
                ResponseSchema(name="怒り", description="an integer between 0-5"),
                ResponseSchema(name="自責志向",description="an integer between 0-5"),
                ResponseSchema(name="恐れ", description="an integer between 0-5"),
                ResponseSchema(name="自己嫌悪", description="an integer between 0-5"),
                ResponseSchema(name="恥",description="an integer between 0-5"),
                ResponseSchema(name="罪悪感",description="an integer between 0-5"),
                ResponseSchema(name="嫉妬", description="an integer between 0-5"),
                ResponseSchema(name="いらだたしさ",description="an integer between 0-5"),
                ResponseSchema(name="恥ずかしさ",description="an integer between 0-5"),
                ResponseSchema(name="憤り", description="an integer between 0-5"),
                ResponseSchema(name="他人に迷惑をかける心配",description="an integer between 0-5"),
            ]
            return response_schemas
        elif i == 2:
            response_schemas = [
                ResponseSchema(name="人の気持ちに自分が与える影響に感じる責任",description="an integer between 0-4"),
                ResponseSchema(name="自分の気持ちの動きに感じる責任",description="an integer between 0-4"),
                ResponseSchema(name="周りに影響を与えたり変えたりしたい欲求", description="an integer between 0-4")
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

logging.basicConfig(filename=fname+"_jp.log", encoding='utf-8', level=logging.DEBUG)
for i in range(1): # choose n for how many times to run this survey
    df = new.generate_response()
    df.to_csv("results/"+fname+"/jp_"+fname+str(i))
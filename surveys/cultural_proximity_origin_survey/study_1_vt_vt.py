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
        self.culture = "vietnamese"
        self.model = self.__create_model(model_name)
        self.situations = {"Ban thấy vui vì nhận được đánh giá xuất sắc về hiệu suất công việc và được thăng chức. \
                            Nhưng đồng nghiệp bị cảnh cáo do làm việc kém hiệu quả, khiến bạn cảm thấy buồn vui lẫn lộn.": "self-success",\
                            "Sau một trận đấu đầy thử thách, bạn cảm thấy vui sướng sau khi giành vị trí thứ nhất trong một giải đấu. \
                            Trong khi đó, đồng đội của bạn chật vật và kết thúc ở vị trí cuối. \
                            Điều này khiến bạn cảm thấy vui buồn lẫn lộn giữa vui mừng và cảm thông.": "self-success",\
                            "Bạn cảm thấy vui sướng sau khi bạn thử vai diễn cho một vở kịch và nhận được vai chính.\
                             Ngược lại, bạn của bạn chỉ nhận được vai thứ yếu. \
                            Bạn cảm thấy lẫn lộn giữa niềm vui và sự đồng cảm.": "self-success",\
                            "Trong một dự án làm việc nhóm, bạn cảm thấy đặc biệt vui sướng khi nhận được điểm A. \
                            Nhưng một thành viên trong nhóm chỉ nhận được điểm thấp. Điều này khiến bạn cảm thấy lẫn\
                            lộn giữa tự hào và lo lắng cho kết quả chung của cả nhóm.": "self-success",\
                            "Bạn cảm thấy thành công khi các tác phẩm nghệ thuật của bạn được khen ngợi rộng rãi và được bán chạy ở một buổi triển lãm.\
                             Ngược lại, nghệ sĩ đồng nghiệp của bạn chật vật để lôi kéo sự chú ý cho tác phẩm của họ, \
                            điều này khiến bạn cảm thấy lẫn lộn giữa tự hào và cảm thông.": "self-success",\
                            "Bạn cảm thấy tự hào khi bạn thân của bạn thi đấu xuất sắc trong một giải thể thao và giành vị trí thứ nhất.\
                             Tuy nhiên, bạn thi đấu không tốt, \
                            khiến bạn cảm thấy lẫn lộn giữa niềm vui cho bạn thân và sự thất vọng cho bản thân mình.": "self-failure",\
                            "Bạn cảm thấy tự hào khi tác phẩm của anh họ được trưng bày trong một triển lãm danh tiếng, \
                            và được ngợi khen nức nở. Trong khi đó, bạn chật vật để thu hút của sự chú ý cho các tác phẩm của mình. \
                            Điều này khiến bạn cảm thấy lẫn lộn giữa khâm phục cho sự thành công của anh họ, \
                            và thất vọng với sự tiến triển của mình.": "self-failure",\
                            "Bạn cảm thấy tự hào khi bạn thân nhận được một công việc trong một công ty hàng đầu. \
                            Trong khi đó, bạn đối diện với sự thụt lùi trong sự nghiệp. \
                            Điều này khiến bạn cảm thấy lẫn lộn giữa niềm vui cho bạn thân và sự thất vọng cho sự nghiệp của mình.": "self-failure", \
                            "Bạn cảm thấy tự hào khi bạn thân của mình là tâm điểm của sự chú ý trong các buổi hội họp xã giao, dễ dàng kết bạn với mọi người.\
                             Trong khi đó, bạn chật vật để theo kịp tình huống xã giao và cảm thấy bị bỏ lại. \
                            Điều này khiến bạn cảm thấy lẫn lộn giữa tự hào cho bạn thân và thất vọng cho bản thân.": "self-failure",
                            "Bạn cảm thấy tự hào khi hàng xóm của bạn nhận được đề cử cho thành tích học vấn như là giành được học bổng hoặc\
                             được vinh danh thủ khoa. Tuy nhiên, bạn cảm thấy thất vọng khi bạn so sánh thành tích học tập với họ. \
                            Điều này khiến bạn cảm thấy lẫn lộn giữa tự hào và tự ti.": "self-failure"}
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
                    att_list = ['Cảm xúc tích cực', 'Cảm xúc tiêu cực']
                    i=0
                    while(i<self.sample):
                        logging.info('in the while loop')
                        try:
                            response = self.chain0.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 4)
                            if ok:
                                positive.append(int(response['Cảm xúc tích cực']))
                                negative.append(int(response['Cảm xúc tiêu cực']))
                                logging.info('here')
                                i+=1
                        except:
                            pass
                if j == 1:
                    happiness, pride, sympathy, relief, hope, friendly_feeling,\
                    sadness, anxiety, anger, self_blame, fear, anger_at_oneself, shame, guilt, jealousy,\
                    frustration, embarrassment, resentment, troubling_someone = ([] for i in range(19))
                    att_list = ['Vui vẻ', 'Tự hào', 'Cảm thông', 'Khuây khỏa', 'Hi vọng', 'Thân thiện', \
                                'Buồn', 'Lo lắng', 'Giận dữ', 'Tự đổ lỗi', 'Sợ hãi', 'Giận dữ người khác', \
                                'Xấu hổ', 'Cảm thấy có lỗi', 'Ghen tị', 'Thất vọng', 'Ngượng ngùng', 'Cay đắng', \
                                'Sợ gây rắc rối cho người khác']
                    i=0
                    while(i<self.sample):
                        logging.info('in the second while loop')
                        try:
                            response = self.chain1.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 5)
                            if ok:
                                logging.info('ok second response')
                                happiness.append(response['Vui vẻ'])
                                pride.append(response['Tự hào'])
                                sympathy.append(response['Cảm thông'])
                                relief.append(response['Khuây khỏa'])
                                hope.append(response['Hi vọng'])
                                friendly_feeling.append(response['Thân thiện'])

                                sadness.append(response['Buồn'])
                                anxiety.append(response['Lo lắng'])
                                anger.append(response['Giận dữ'])
                                self_blame.append(response['Tự đổ lỗi'])
                                fear.append(response['Sợ hãi'])

                                anger_at_oneself.append(response['Giận dữ người khác'])
                                shame.append(response['Xấu hổ'])
                                guilt.append(response['Cảm thấy có lỗi'])
                                jealousy.append(response['Ghen tị'])

                                frustration.append(response['Thất vọng'])
                                embarrassment.append(response['Ngượng ngùng'])
                                resentment.append(response['Cay đắng'])
                                troubling_someone.append(response['Sợ gây rắc rối cho người khác'])
                                i+=1
                        except:
                            pass
                if j == 2:
                    influence, res4others, res4me = ([] for i in range(3))
                    att_list = ['Có trách nhiệm cho cảm xúc của người khác', 'Có trách nhiệm cho cảm xúc bản thân',\
                                 'Ham muốn tác động và thay đổi những người quanh mình']
                    i=0
                    while(i<self.sample):
                        logging.info('in the third while loop')
                        try:
                            response = self.chain2.invoke({"situation": situation})
                            ok = self.__output_validation(response, att_list, 0, 4)
                            if ok:
                                logging.info('ok third response')
                                res4others.append(response['Có trách nhiệm cho cảm xúc của người khác'])
                                res4me.append(response['Có trách nhiệm cho cảm xúc bản thân'])
                                influence.append(response['Ham muốn tác động và thay đổi những người quanh mình'])
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
            return PromptTemplate(template="Chúng tôi đánh giá cao sự tham gia của bạn. \
                                            Sau đây là khảo sát về văn hóa và cảm xúc. \
                                            Xin hãy đánh giá tình huống sau dựa theo cảm xúc tích cực và cảm xúc tiêu cực của bạn theo thang điểm 5 với nhãn từ 0 (không cảm thấy gì) đến 4 (cảm thấy sâu sắc).\
                                            Mặc dù tình huống có thể phản ánh cảm xúc của người viết, \
                                            với tư cách là một người tham gia, bạn có thể không cảm thấy giống như vậy. \
                                            Vui lòng đánh giá điều này với tư cách là một người tham gia đến từ Việt Nam.\
                                  \n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})
        if i == 1:
            return PromptTemplate(template="Chúng tôi đánh giá cao sự tham gia của bạn. \
                                            Sau đây là khảo sát về văn hóa và cảm xúc.\
                                            Xin hãy đánh giá tình huống sau dựa theo các cảm xúc tích cực cụ thể như vui vẻ, tự hào, cảm thông, khuây khỏa,\
                                            hi vọng, thân thiện; cùng với những cảm xúc tiêu cực cụ thể như buồn, lo lắng, giận dữ, tự đổ lỗi, sợ hãi, giận dữ người khác,\
                                            xấu hổ, cảm thấy có lỗi, ghen tị, thất vọng, người ngùng, cay đắng, và sợ gây rắc rối cho người khác, \
                                            theo thang điểm 6 với nhãn từ 0 (không cảm thấy gì) đến 5 (cảm thấy sâu sắc). Mặc dù tình huống có thể phản ánh cảm xúc của người viết, \
                                            với tư cách là một người tham gia, bạn có thể không cảm thấy giống như vậy. Vui lòng đánh giá điều này với tư cách là một người tham gia đến từ Việt Nam.\
                                  \n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})
        if i == 2:
            return PromptTemplate(template="Chúng tôi đánh giá cao sự tham gia của bạn. \
                                            Sau đây là khảo sát về văn hóa và cảm xúc. \
                                            Xin hãy đánh giá tình huống sau dựa trên mức độ mà bạn chịu trách nhiệm cho cảm xúc của người khác, \
                                            mức độ mà người khác chịu trách nhiệm về cảm xúc của bạn và cuối cùng mức độ mà  bạn nghĩ về ảnh hưởng hoặc thay đổi những người,\
                                            sự kiện, đối tượng xung quanh dựa theo ý muốn riêng của bạn, sử dụng thang điểm 5 với nhãn từ 0 (không cảm thấy gì) đến 4 (cảm thấy sâu sắc).\
                                            Mặc dù tình huống có thể phản ánh cảm xúc của người viết, với tư cách là một người tham gia, bạn có thể không cảm thấy giống như vậy. Vui lòng đánh giá điều này với tư cách là một người tham gia đến từ Việt Nam.\
                                  \n{format_instructions}\n{situation}",\
                    input_variables=["situation"],
                    partial_variables={"format_instructions": format_instructions})


    def __create_schemas(self, i):
        if i == 0:
            response_schemas = [
                ResponseSchema(name="Cảm xúc tích cực", description="an integer between 0-4"),
                ResponseSchema(name="Cảm xúc tiêu cực",description="an integer between 0-4"),
            ]
            return response_schemas
        elif i == 1:
            response_schemas = [
                ResponseSchema(name="Vui vẻ", description="an integer between 0-5"),
                ResponseSchema(name="Tự hào",description="an integer between 0-5"),
                ResponseSchema(name="Cảm thông", description="an integer between 0-5"),
                ResponseSchema(name="Khuây khỏa",description="an integer between 0-5"),
                ResponseSchema(name="Hi vọng", description="an integer between 0-5"),
                ResponseSchema(name="Thân thiện",description="an integer between 0-5"),
                ResponseSchema(name="Buồn", description="an integer between 0-5"),
                ResponseSchema(name="Lo lắng",description="an integer between 0-5"),
                ResponseSchema(name="Giận dữ", description="an integer between 0-5"),
                ResponseSchema(name="Tự đổ lỗi",description="an integer between 0-5"),
                ResponseSchema(name="Sợ hãi", description="an integer between 0-5"),
                ResponseSchema(name="Giận dữ người khác", description="an integer between 0-5"),
                ResponseSchema(name="Xấu hổ",description="an integer between 0-5"),
                ResponseSchema(name="Cảm thấy có lỗi",description="an integer between 0-5"),
                ResponseSchema(name="Ghen tị", description="an integer between 0-5"),
                ResponseSchema(name="Thất vọng",description="an integer between 0-5"),
                ResponseSchema(name="Ngượng ngùng",description="an integer between 0-5"),
                ResponseSchema(name="Cay đắng", description="an integer between 0-5"),
                ResponseSchema(name="Sợ gây rắc rối cho người khác",description="an integer between 0-5"),
            ]
            return response_schemas
        elif i == 2:
            response_schemas = [
                ResponseSchema(name="Có trách nhiệm cho cảm xúc của người khác",description="an integer between 0-4"),
                ResponseSchema(name="Có trách nhiệm cho cảm xúc bản thân",description="an integer between 0-4"),
                ResponseSchema(name="Ham muốn tác động và thay đổi những người quanh mình", description="an integer between 0-4")
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

logging.basicConfig(filename=fname+"_vt.log", encoding='utf-8', level=logging.DEBUG)
for i in range(30):
    df = new.generate_response()
    df.to_csv("results/"+fname+"/vt_"+fname+str(i))
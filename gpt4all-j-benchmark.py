from pygpt4all import GPT4All_J
#from transformers import AutoTokenizer
import nltk
import time
import os
import sys
import csv
import random
import argparse

# Help message
HELP  = "---------------------------------------------------------------------------------------------\n"
HELP += "==  LLM Model Benchmark on CPU                                                             ==\n"
HELP += "==                                                                                         ==\n"
HELP += "==  Usage:                                                                                 ==\n"
HELP += "==          -n x: Set n_predict parameter for the LLM model API                            ==\n"
HELP += "==          -t x: Set CPU logical threads number to run the LLM model                      ==\n"
HELP += "==          -c x: Set number of questions randomly from the full question list             ==\n"
HELP += "==          -a 0/1: Allow user to halt the program at the very beginning                   ==\n"
HELP += "==          -o xxx: Set output CSV log path                                                ==\n"
HELP += "---------------------------------------------------------------------------------------------\n"

# Question pool
listQuestion = [
    "What is the capital of France?",
    "Briefly explain what a blockchain is.",
    "Explain the theory of evolution in your own words.",
    "Do you think artificial intelligence will eventually surpass human intelligence? Why or why not?",
    "Compare and contrast the major political parties in the United States. Discuss their main policy positions and histories.",
    "Select one of the USA sustainable development goals and explain how it could be achieved through government policies or private sector innovation.",
    "Discuss the pros and cons of social media platforms like Facebook, Instagram, and Twitter on both individuals and society as a whole.",
    "Describe what life was like 100 years ago. Consider areas such as technology, jobs, family life, entertainment, and education.",
    "You have won an all-expenses paid trip to one location of your choosing. Where would you go and why? Describe what you would choose to do during your trip.",
    "It was the best of times, it was the worst of times. Elaborate on what you think this quote means and how it might apply today. Discuss both optimistic and pessimistic interpretations of the state of the modern world.",
    "What do you think are some of the biggest challenges facing young people today in terms of opportunities for career and personal fulfillment? Discuss how both technology and globalization might impact future job prospects and societal expectations placed on the next generation.",
    "Discuss how the rise of online media platforms have influenced both the production and consumption of news. Consider factors such as tailored content, spread of misinformation, competition for advertising revenue, and decline in print media. Speculate how news reporting may evolve over the next decade.",
    "Compare the American political system with another political system of your choosing. Discuss factors such as voter participation rates, role of political parties, campaign finance laws, and frequency of elections. Analyze strengths and weaknesses of the systems and how they might influence policy outcomes.",
    "Select one controversial emerging biotechnology, such as genetic engineering or artificial organs. Discuss the technological, scientific and ethical considerations surrounding this technology. Identify key issues at stake and arguments on both sides. Provide your view on how to ensure responsible development and use of this technology. ",
    "Compare the roles that family and community play in individual wellbeing across different societies and time periods. For example, contrast more traditional versus modern societies, or contrast different generations within the same society. Discuss how the meaning of concepts like happiness, success, and fulfillment can take on different forms based on cultural and environmental influences."
]

# Default parameters
defaultCountThread = 16
defaultNPredict = 1024
defaultSeed = 0
defaultTopK = 40
defaultTopP = 0.9
defaultTemperature = 0.1
defaultFileLog = "benchmark_GPT4ALLJ.csv"
defaultDebug = 0


# Log function, add a line each time
def appendCsv(file, row):
    try:
        with open(file, mode='a', newline='', encoding='utf-8-sig') as f:
            write = csv.writer(f)
            write.writerow(row)
            f.close
    except:
        pass

# Get token count
'''
def getTokenCount(text):
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))

    print(os.path.join(application_path, "model/bert-base-cased"))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(application_path, "model/bert-base-cased"))
    tokens = tokenizer.tokenize(text)
    return len(tokens)
'''

def getTokenCount(text):
    nltk.data.path.append("model/nltk_data")
    tokens = nltk.word_tokenize(text)
    return len(tokens)

# Arg function, pass in args to main function
def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npredict", "-n", type=int, default=defaultNPredict, help="n_predict parameter for model API")
    parser.add_argument("--threads", "-t", type=int, default=defaultCountThread, help="CPU threads count parameter for model API")
    parser.add_argument("--count", "-c", type=int, default=0, help="randomly pick x questions from the full question list")
    parser.add_argument("--halt", "-a", type=int, default=0, help="wait at the beginning for debug/analysis purpose")
    parser.add_argument("--output", "-o", type=str, default=defaultFileLog, help="assign the output log file path")
    args = parser.parse_args()
    return args

# Main function.
if __name__ == "__main__":
    # Help message
    print(HELP)

    # Receive user input parameters
    args = getArgs()
    nPredict = args.npredict
    countThread = args.threads
    countQuestions = args.count
    enableHalt = args.halt
    fileLog = args.output

    if nPredict == 0:   # if user didn't input anything, then the default value 0 should convert to None, means no limitation
        nPredict = None

    if countQuestions == 0 or countQuestions > len(listQuestion):
        listQuestionProceed = listQuestion
    else:
        listQuestionProceed = random.sample(listQuestion, countQuestions)
        random.shuffle(listQuestionProceed)

    print()
    print("==== N_Predict: ", nPredict)
    print("==== Threads: ", countThread)
    print("==== Questions: ", countQuestions)
    print("==== Enable halt: ", enableHalt)
    print("==== Log file: ", fileLog)
    print()

    if enableHalt == 1:
        input("**** Halt for you to adjust system config, press Enter to continue ****")

    # Create log file if not exists, and the first line
    if not os.path.exists(fileLog):
        rowFirstLine = ["Time Stamp", "Threads", "Tokens Input", "Tokens Output", "Time Cost 1st Token", "Time Cost Overall", "Latency 1st Token", "Latency Average", "Tokens per Second", "Index", "Question"]
        appendCsv(fileLog, rowFirstLine)

    countInputTokenAll = 0
    countOutputTokenAll = 0   # total tokens output during Q&A outputs
    countQuestionAll = 0
    timeSpendFirstAll = 0 # total time for 1st token
    timeSpendOverallAll = 0    # total time spend
    model = GPT4All_J('model/ggml-gpt4all-j-v1.3-groovy.bin')
    # Go through questions in the list
    for index, question in enumerate(listQuestionProceed):
        countInputToken = getTokenCount(question)
        countOutputToken = 0  # totkens for current Q&A output
        timeStart = time.time() # time stamp for current Q&A output, start
        answer = "" # answer initialized as null string
        if defaultDebug != 0:
            print(question)
        for token in model.generate(question+"\n\n", n_predict=nPredict, n_threads=countThread, seed=defaultSeed, top_k=defaultTopK, top_p=defaultTopP, temp=defaultTemperature):    # fix seed to ensure generate same answer string for same question
            if defaultDebug != 0:
                print(token, end='', flush=True)
            if countOutputToken == 0:
                timeFirst = time.time()
            answer += token
            countOutputToken += 1
        timeEnd = time.time()   # time stamp for current Q&A output, end
        timeSpendFirst = timeFirst - timeStart
        timeSpendOverall = timeEnd - timeStart
        countInputTokenAll += countInputToken
        countOutputTokenAll += countOutputToken
        countQuestionAll += 1
        timeSpendFirstAll += timeSpendFirst
        timeSpendOverallAll += timeSpendOverall
        # output to command line for more info
        print()
        print("*"*100)
        print("==== Question: ")
        print("---- ", question)
        print("==== Answer: ")
        print("---- ", answer)
        print("==== Input Tokens: ")
        print("---- ", str(countInputToken))
        print("==== Output Tokens: ")
        print("---- ", str(countOutputToken))
        print("==== Time 1st token: ")
        print("---- ", str(timeSpendFirst))
        print("==== Time for whole Question: ")
        print("---- ", str(timeSpendOverall))
        # record the data for current Q&A
        rowCurrent = [time.strftime("%Y/%m/%d-%H:%M:%S", time.localtime()), str(countThread), str(countInputToken), str(countOutputToken), format(timeSpendFirst, '.2f'), format(timeSpendOverall, '.2f'), format(timeSpendFirst*1000/1, '.2f'), format((timeSpendOverall-timeSpendFirst)*1000/(countOutputToken-1), '.2f'), format((countOutputToken-1)/(timeSpendOverall-timeSpendFirst), '.2f'), str(index), question]
        appendCsv(fileLog, rowCurrent)
        model.reset()
    # record the data for all Q&A from list
    rowAll = [time.strftime("%Y/%m/%d-%H:%M:%S", time.localtime()), str(countThread), str(countInputTokenAll), str(countOutputTokenAll), format(timeSpendFirstAll, '.2f'), format(timeSpendOverallAll, '.2f'), format(timeSpendFirstAll*1000/countQuestionAll, '.2f'), format((timeSpendOverallAll-timeSpendFirstAll)*1000/(countOutputTokenAll-countQuestionAll), '.2f'), format((countOutputTokenAll-countQuestionAll)/(timeSpendOverallAll-timeSpendFirstAll), '.2f'), "SUM", "-"]
    appendCsv(fileLog, rowAll)
    appendCsv(fileLog, [])

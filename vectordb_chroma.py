# Job Listing using Chroma Vector to chuck the charcter from a document and to find matching job skill from input of the user with chroma 
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

document = TextLoader("job_listings.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks = text_splitter.split_documents(document)
db = Chroma.from_documents(chunks, llm)

text = input("Enter the query")
embedding_vector = llm.embed_query(text)
docs = db.similarity_search_by_vector(embedding_vector)

for doc in docs:
  print(doc.page_content)

# another method  
# retriever = db.as_retriever()
# text = input("Enter the Query")
# for doc in docs:
#    print(doc.page_content)

# input Job_listing.txt
# 1. Software Engineer at TechCorp - Responsibilities include developing and maintaining software applications, collaborating with cross-functional teams, and ensuring code quality. Requires proficiency in Java, Python, and SQL.
# 2. Data Scientist at DataMinds - Duties involve analyzing large datasets, building predictive models, and presenting insights to stakeholders. Requires expertise in Python, R, and machine learning.
# 3. Digital Marketing Specialist at MarketGurus - Role includes creating and managing online marketing campaigns, analyzing web traffic, and optimizing SEO. Requires experience with Google Analytics, SEM, and content creation.
# 4. Project Manager at BuildIt - Responsibilities include overseeing construction projects, managing budgets, and coordinating with contractors. Requires strong leadership skills and knowledge of project management software.
# 5. Graphic Designer at CreativeWorks - Role involves designing marketing materials, collaborating with the creative team, and adhering to brand guidelines. Requires proficiency in Adobe Creative Suite and a strong portfolio.
# 6. Financial Analyst at FinExperts - Duties include analyzing financial data, preparing reports, and advising on investment decisions. Requires strong analytical skills and experience with financial modeling.
# 7. Human Resources Manager at PeopleFirst - Responsibilities include recruiting, onboarding, and managing employee relations. Requires excellent communication skills and knowledge of HR software.
# 8. Cybersecurity Specialist at SecureNet - Role involves protecting the company's IT infrastructure, monitoring for security breaches, and implementing security protocols. Requires expertise in network security and experience with security tools.
# 9. Sales Manager at RetailStars - Duties include managing the sales team, developing sales strategies, and achieving sales targets. Requires strong leadership skills and experience in retail sales.
# 10. Content Writer at WordSmiths - Responsibilities include creating engaging content for blogs, social media, and websites. Requires excellent writing skills and a creative mindset.

# Output Enter your Query Digitial Marketing 
# 3. Digital Marketing Specialist at MarketGurus - Role includes creating and managing online marketing campaigns, analyzing web traffic, and optimizing SEO. Requires experience with Google Analytics, SEM, and content creation.
# 5. Graphic Designer at CreativeWorks - Role involves designing marketing materials, collaborating with the creative team, and adhering to brand guidelines. Requires proficiency in Adobe Creative Suite and a strong portfolio.
# 10. Content Writer at WordSmiths - Responsibilities include creating engaging content for blogs, social media, and websites. Requires excellent writing skills and a creative mindset.



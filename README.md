## The Implementation of Image Similarity Search using Vector Search
Similar image search is a notable application within the realm of Recommendation Systems. In this blog, I will provide an in-depth exploration of the step-by-step process involved in constructing a Similar Image Search System. Notably, I will highlight the utilization of the cutting-edge Vector Search engine recently introduced by the MongoDB team, showcasing its capabilities and contributions to the field.
![](./resources/search_results.png)

## Deployment

We can imagine the step by step to bring deep learning from the training phase into the inference phase:

1. Choosing the gallery image set, and then using the model to extract the representative features of this set. 
2. For every time, there is a new image as the query, we also need to present it in the feature space.
3. Comparing the distance between the query feature and the gallery features. Two of the most popular approaches are Euclidean distance and Cosine Similarity.

However, the feature extraction step is time-consuming and needs a lot of computational cost. We cannot keep the massive feature volume in the memory for a long time to wait for the query image.

In the initial quest to tackle this challenge, the Facebook research team developed the FAISS library. While FAISS has made significant contributions, there are alternative approaches available to address the same problem, including Elastic Search and Milvus. Each library comes with its own advantages and limitations. By conducting experiments and gaining insights from these tools, you can explore their capabilities and choose the most appropriate solution for your specific problem.

Within the scope of this blog, I am delighted to delve into Vector Search, one of the five exciting new capabilities recently introduced by MongoDB at the end of last month. Vector Search opens up fresh possibilities for advanced search and retrieval techniques, leveraging the power of vectors to enhance the efficiency and accuracy of search operations. Let's explore this remarkable feature and its potential impact on modern applications.

### MongoDB Atlas

MongoDB Atlas, introduced in 2016, is a cloud-native, fully managed database service that enables developers to accelerate application development. Since its launch, Atlas has evolved into a comprehensive developer data platform, supporting various workloads such as transactions, search, analytics, and streaming. It offers a battle-tested infrastructure, ensuring high availability, scalability, and top-tier security. Excitingly, MongoDB Atlas now includes native support for vectors through Atlas Vector Search, simplifying the development of next-generation applications.

### Vector Search

Atlas Vector Search, an integral part of MongoDB Atlas, eliminates the need to duplicate or transform your data, learn new syntax, or manage additional infrastructure. It enables you to seamlessly store vector embeddings alongside your operational data. With the flexibility provided by Atlas Triggers, your vector entries can be dynamically updated when your data changes. Furthermore, your application only needs to interact with a unified query interface, simplifying the integration of vector search functionality. By leveraging these powerful capabilities within the robust MongoDB Atlas platform, you can accelerate application development without compromising on performance or reliability.

### Setup the Vector Search

Follow the below steps to set up a database:

1. Create a cluster in MongoDB Atlas
2. Then access the cluster and create a new project
![](./resources/atlas_1.png)
3. The next thing is to access the project and create a new database and collection. In this blog, I will create a database called clothes_search and a collection named feature512
4. To connect the database from your computer, we need to install the Python package of MongoDB.
```bash
pip install pymongo
```
![](./resources/atlas_2.png)
After installing the pymongo, let’s back the to database dashboard and click on Connect to obtain the connection string. Then choose the connection modal is Drivers and take the examples.
![](./resources/atlas_3.png)
Now, we can access Atlas from your local machine by using this script
```python
import pymongo
client = pymongo.MongoClient(os.getenv("mongodb+srv://dev01:abc13579@cluster0.vmpt3h1.mongodb.net/?retryWrites=true&w=majority"))
db = client.clothes_search
collection = db['feature512']
```
### Insert the features into DB

As mentioned above, we need to choose a gallery set and get their feature representations. After obtaining the features, we use the client’s Mongo API to insert them into the database intermediately.
```python
import torch.nn.functional as F
def insert_feature_to_db(
        collection, 
        model, 
        dataloader, 
        device=torch.device('cpu')
    ):
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, img_info in tqdm(dataloader, total=len(dataloader)):
            images = images.to(device, non_blocking=True)
            outputs = F.normalize(model(images))
            outputs = outputs.detach().cpu().numpy()
            for img_info, output in zip(info_imgs, outputs):
                collection.insert_one(
                    {
                        "image_name": img_info,
                        "embedding": output.tolist()
                    }
                )
```

The results insertion are shown in the below figure
![](./resources/atlas_4.png)
### Setup the Search Index and Querying

Before starting to search for anything you want, let’s just configure the method that you will use to communicate with the database. In the above figure, let’s click on the Search Indexes and create the new Index
![](./resources/atlas_5.png)
Should choose the term JSON Editor to start creating the new one. To search for a highly-dimensional vector, let’s follow my configuration.
```json
{
  "mappings": {
    "fields": {
      "embedding": [
        {
          "dimensions": 512,
          "similarity": "cosine",
          "type": "knnVector"
        }
      ]
    }
  }
}
```
You need to pay attention to three terms including:

1. “embedding” corresponds to the field name that you want to compare in the database.
2. “dimensions” must match the initial dimension of the feature representations.
3. “similarity” implies the method that we will use to compare two features.

Finally, you can search for anything that you want by using a short Python script

```python
img = read_image(img_path, target_size=128)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = ModelEmbedding(config = cfg)
ckpt = torch.load(
    os.path.join(cfg.SAVE_DIR, "checkpoint.pth"),
    map_location="cpu"
)
model.load_state_dict(ckpt['model'])
model.eval()
model.to(device)
img.to(device)
with torch.no_grad():
    vector_query = F.normalize(model(img),1)[0].detach(
        ).cpu().numpy().tolist()
    pipeline = [
        {
            "$search": {
                "index": "feature512",
                "knnBeta": {
                    "vector": vector_query,
                    "path": "embedding",
                    "k": 10
                }
            }
        },
        {
            "$project": {
                "embedding": 0,
                "_id": 0,
                'score': {
                    '$meta': 'searchScore'
                }
            }
        },
    ]
    res = collection.aggregate(pipeline)
    list_sim = []
    for r in res:
        list_sim.append((r['image_name'], r['score']))
```

## Results

To assess the performance of both the deep learning model and the search method of Atlas Vector Search, I employed external images that were not part of the database. This enabled me to thoroughly evaluate the effectiveness and robustness of the framework I built. As you can see in the figure, the above item is the query image and the ten below images are the most similar samples in the gallery.

## Conclusion

Throughout this blog, I had the pleasure of introducing you to a remarkable and innovative capability of MongoDB: the Vector Search tool. This powerful feature empowers developers to explore advanced search and retrieval techniques, leveraging the potential of vectors to enhance the efficiency and accuracy of queries. With the Vector Search tool, MongoDB opens up exciting possibilities for applications in various domains such as recommendation systems, similarity matching, and content-based search. By seamlessly integrating this tool into your MongoDB-powered projects, you can unlock new dimensions of data exploration and deliver enhanced user experiences.
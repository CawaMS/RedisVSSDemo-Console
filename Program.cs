using StackExchange.Redis;
using NRedisStack;
using NRedisStack.RedisStackCommands;
using NRedisStack.Search;
using NRedisStack.Search.Literals.Enums;
using Microsoft.Extensions.Configuration;
using static NRedisStack.Search.Schema;
using Azure;
using Azure.AI.OpenAI;

//add user secret config
var config = new ConfigurationBuilder().AddUserSecrets<Program>().Build();

string redisConnection = config["redisConnection"];
//<cache_name>.eastus.redisenterprise.cache.azure.net:10000,password=<primary_access_key>,ssl=True,abortConnect=False
string aoaiConnection = config["aoaiConnection"];

ConnectionMultiplexer redis = ConnectionMultiplexer.Connect(redisConnection);
IDatabase db = redis.GetDatabase();

//https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/embeddings?tabs=csharp
// initialize Azure open ai ada-text-embeddings service
Uri aoaiEndpoint = new(aoaiConnection);
string aoaiKey = config["aoaiKey"];

string embeddingsDeploymentName = config["textEmbeddingsDeploymentName"];

AzureKeyCredential credentials = new(aoaiKey);

OpenAIClient openAIClient = new OpenAIClient(aoaiEndpoint, credentials);

// initialize data in Redis with vector embeddings on the descriptions
db.HashSet("id:1",
[
    new("Name", "Top-handle"),
    new("Price",77),
    new("Brand", "CathyDesign"),
    new("Category", "Purse" ),
    new("description","A purse with top handle. Multiple colors avaiable. Suitable for occasions such as going to the office, weekends hang-outs, going out for dinners, and parties.\r\n"),
    new("description_embeddings", textToEmbeddings("A purse with top handle. Multiple colors avaiable. Suitable for occasions such as going to the office, weekends hang-outs, going out for dinners, and parties.\r\n",
                                                  openAIClient, embeddingsDeploymentName).SelectMany(BitConverter.GetBytes).ToArray())
]);

db.HashSet("id:2",
[
    new("Name", "Boots"),
    new("Price",160),
    new("Brand", "LapinArt"),
    new("Category", "Shoes" ),
    new("description","Vegan-leather boots. Multiple colors available. Suitable to wear in spring and autumn. Suitable to both formal and casual occasions."),
    new("description_embeddings", textToEmbeddings("Vegan-leather boots. Multiple colors available. Suitable to wear in spring and autumn. Suitable to both formal and casual occasions.",
                                                   openAIClient, embeddingsDeploymentName).SelectMany(BitConverter.GetBytes).ToArray())
]);

db.HashSet("id:3",
[
    new("Name", "speedy"),
    new("Price",245),
    new("Brand", "LapinArt"),
    new("Category", "Purse" ),
    new("description","A purse with top handle and cross-body straps. Only one color available. Suitable for occasions such as going to the office, weekends hang-outs, shopping, and parties.\r\n"),
    new("description_embeddings", textToEmbeddings("A purse with top handle and cross-body straps. Only one color available. Suitable for occasions such as going to the office, weekends hang-outs, shopping, and parties.\r\n",
                                                   openAIClient, embeddingsDeploymentName).SelectMany(BitConverter.GetBytes).ToArray())
]);

db.HashSet("id:4",
[
    new("Name", "Dressing"),
    new("Price",120),
    new("Brand", "CathyDesign"),
    new("Category", "Shoes" ),
    new("description","Vegan-leather dressing shoes. Only one color available. Suitable to wear all seasons. Suitable to formal occasions."),
    new("description_embeddings", textToEmbeddings("Vegan-leather dressing shoes. Only one color available. Suitable to wear all seasons. Suitable to formal occasions.",
                                                    openAIClient, embeddingsDeploymentName).SelectMany(BitConverter.GetBytes).ToArray())
]);

db.HashSet("id:5",
[
    new("Name", "Messenger"),
    new("Price",229),
    new("Brand", "LapinArt"),
    new("Category", "Purse" ),
    new("description","A purse with cross-body straps. Multiple colors available. Suitable for casual occasions."),
    new("description_embeddings", textToEmbeddings("A purse with cross-body straps. Multiple colors available. Suitable for casual occasions.",
                                                   openAIClient, embeddingsDeploymentName).SelectMany(BitConverter.GetBytes).ToArray())
]);

db.HashSet("id:6",
[
    new("Name", "Handle"),
    new("Price", 249),
    new("Brand", "LapinArt"),
    new("Category", "Purse" ),
    new("description","A purse with handle. Only one color available. Suitable for traveling in all seasons."),
    new("description_embeddings",textToEmbeddings("A purse with handle. Only one color available. Suitable for traveling in all seasons.",
                                                 openAIClient, embeddingsDeploymentName).SelectMany(BitConverter.GetBytes).ToArray())
]);

db.HashSet("id:7",
[
    new("Name", "Long boots"),
    new("Price", 235),
    new("Brand", "CathyDesign"),
    new("Category", "Shoes" ),
    new("description","Vegan-leather long boots. Multiple colors available. Suitable to wear in autumn and winter. Suitable for formal occasions."),
    new("description_embeddings", textToEmbeddings("Vegan-leather long boots. Multiple colors available. Suitable to wear in autumn and winter. Suitable for formal occasions.",
                                                  openAIClient, embeddingsDeploymentName).SelectMany(BitConverter.GetBytes).ToArray())
]);

SearchCommands ft = db.FT();

float[] _description1 = textToEmbeddings(db.HashGet("id:7", "description").ToString(), openAIClient, embeddingsDeploymentName);

// index each vector field
try { ft.DropIndex("vss_products"); } catch { };
Console.WriteLine("Creating search index in Redis");
Console.WriteLine();
ft.Create("vss_products", new FTCreateParams().On(IndexDataType.HASH).Prefix("id:"),
    new Schema()
    .AddTagField("Name")
    .AddVectorField("description_embeddings", VectorField.VectorAlgo.FLAT,
        new Dictionary<string, object>()
        {
            ["TYPE"] = "FLOAT32",
            ["DIM"] = _description1.Length.ToString(),
            ["DISTANCE_METRIC"] = "L2"
        }
));

// search through the descriptions
var res1 = ft.Search("vss_products",
                    new Query("*=>[KNN 2 @description_embeddings $query_vec]")
                    .AddParam("query_vec", _description1.SelectMany(BitConverter.GetBytes).ToArray())
                    .SetSortBy("__description_embeddings_score")
                    .Dialect(2));

foreach (var doc in res1.Documents)
{
    foreach (var item in doc.GetProperties())
    {
        if (item.Key == "__description_embeddings_score")
        {
            Console.WriteLine($"id: {doc.Id}, score: {item.Value}");
            Console.WriteLine("Item Name: " + db.HashGet(doc.Id, "Name"));
            Console.WriteLine("Item description: " + db.HashGet(doc.Id, "description"));
            Console.WriteLine();
        }
    }
}


float[] textToEmbeddings(string text, OpenAIClient _openAIClient, string embeddingsDeploymentName)
{
    EmbeddingsOptions embeddingOptions = new EmbeddingsOptions()
    {
        DeploymentName = embeddingsDeploymentName,
        Input = { text },
    };

    return _openAIClient.GetEmbeddings(embeddingOptions).Value.Data[0].Embedding.ToArray();
}
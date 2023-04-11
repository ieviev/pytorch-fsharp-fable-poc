#r "nuget: Fable.Python"

// these should be auto-generated with static analysis
module Bindings = 

    module Pytorch = 
        open Fable.Core

        [<AbstractClass>]
        [<Import("Dataset",from="torch.utils.data")>]
        type Dataset() =
           abstract member __len__ : unit -> int
           abstract member __getitem__ : idx:int -> int

        type ILossFunction = 
            [<Emit("$0($1, $2)")>]
            abstract member call : weight:obj * size_average:obj -> obj
            abstract member backward : unit -> obj
            

        [<Import("BCEWithLogitsLoss",from="torch.nn")>]
        type BCEWithLogitsLoss(weight:obj, size_average:obj) =
            member this.backward() = jsNative
        
        [<Import("Adam",from="torch.optim")>]
        type Adam(parameters:obj) =
            member this.zero_grad() = jsNative
            member this.step() = jsNative


        type ITorch =
            [<Emit("$0.nn.BCEWithLogitsLoss()")>]
            abstract BCEWithLogitsLoss: unit -> ILossFunction
            abstract tensor: obj -> obj
            abstract sigmoid: obj -> obj

        [<ImportAll("torch")>]
        let torch: ITorch = nativeOnly
            
    module Transformers = 
        open Fable.Core
        open System

        type ITokenizer = 
            // [<Emit("$0($1...)")>]
            // abstract member call : [<ParamArray>] args:obj[] -> obj[]
            [<Emit("$0($1, padding=$2, truncation=$3, return_tensors=$4)")>]
            abstract member call : args:string seq * ?padding:bool * ?truncation:bool * ?return_tensors:string -> System.Collections.Generic.IDictionary<string,obj>
            [<Emit("$0($1, padding=$2, truncation=$3, return_tensors=$4)")>]
            abstract member call : args:string * ?padding:bool * ?truncation:bool * ?return_tensors:string -> System.Collections.Generic.IDictionary<string,obj>
            [<Emit("$0.encode($1,add_special_tokens=$2)")>]
            abstract member encode : args:string * ?add_special_tokens:bool -> int[]
            abstract member convert_ids_to_tokens : args:int[] -> string[]

        type IModel =
            [<Emit("$0($1...)")>]
            abstract member call : [<ParamArray>] args:obj[] -> obj[]
            abstract member train : args:unit -> unit
            abstract member eval : args:unit -> unit
            abstract member parameters : args:unit -> obj

        [<Import("BertTokenizer",from="transformers")>]
        type BertTokenizer() =
            static member from_pretrained(modelName:string) : ITokenizer = jsNative

        type ITransformers =
            [<Emit("$0.BertForSequenceClassification.from_pretrained($1, num_labels=$2)")>]
            abstract BertForSequenceClassification: modelName:string * num_labels:int -> IModel
            abstract tensor: obj -> obj
            abstract sigmoid: obj -> obj

        [<ImportAll("transformers")>]
        let transformers: ITransformers = nativeOnly

open Fable.Core
open Fable.Core.PyInterop
open Fable.Python.Builtins
open System

open Bindings.Transformers
open Bindings.Pytorch

let print x = printfn "%A" x

let texts = 
    [|
        "positive and negative"; 
        "negative negative negative"; 
        "neutral neutral neutral";
        "just positive label"|]

//3 labels: Positive, Negative, Neutral
let labels = [| 
        [|1; 1; 0|] 
        [|0; 1; 0|] 
        [|0; 0; 1|]
        [|1; 0; 0|]
    |] 

let lossfn = torch.BCEWithLogitsLoss()

[<EntryPoint>] 
let main argv = 

    let tokenizer =
        BertTokenizer.from_pretrained("bert-base-cased")

    let tokens = tokenizer.encode("Hello, I'm a tokenizer",false)
    print (tokenizer.convert_ids_to_tokens(tokens))

    let model = 
        transformers.BertForSequenceClassification("bert-base-cased", num_labels = 3)

    let optimizer = Adam(model.parameters())
    model.train()

    let labels_tensor = torch.tensor(labels)
    let inputs = tokenizer.call(texts, padding=true, truncation=true, return_tensors="pt")
    
    for epoch in 0..9 do
        
        optimizer.zero_grad()
        let outputs = model.call(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
        let loss = lossfn.call(outputs?logits, labels_tensor?float())
        loss?backward()
        printfn "epoch: %i, loss %A" epoch loss
        optimizer.step()
    
    printfn "training done\n\n"

    model.eval()

    let test_text = [|"negative negative negative"|]
    let test_input = tokenizer.call(test_text, padding=true, truncation=true, return_tensors="pt")
    let test_output = model.call(test_input["input_ids"], test_input["attention_mask"], test_input["token_type_ids"])
    let test_logits = test_output?logits
    let test_predictions = torch.sigmoid(test_logits)
    print("test input: negative negative negative")
    print("labels: [positive, negative, neutral]")
    print(test_predictions)
    

    0


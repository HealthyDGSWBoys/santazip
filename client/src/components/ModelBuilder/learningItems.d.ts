export interface learningItems {
    sequential: Array<{
        name: "dropout" | "dense",
        units: number,
        activation: "relu" | "softmax" | "sigmoid" | "linear"
    }>
    processer: Array<"distance" | "angle" | "distance2">
    optimizer: "Adam" | "SGD" | "RMSprop"
    loss: "SparseCategoricalCrossentropy"
    metrics: Array<string>
    epochs: number
    batch: number
}
using System;
using MachineLearning.Models;
using Microsoft.ML;

namespace MachineLearning.ML;

public class CasaModelPredictor
{
    private MLContext mlContext = new MLContext();
    private ITransformer modeloCarregado;
    public void CarregarModelo(string path)
    {
        DataViewSchema modeloSchema;
        modeloCarregado = mlContext.Model.Load(path, out modeloSchema);
    }

    public CasaPredictionResult Prever(CasaInputData novaCasa)
    {
        var predEndgine = mlContext.Model.CreatePredictionEngine<CasaInputData, CasaPredictionResult>(
            modeloCarregado
        );

        return predEndgine.Predict(novaCasa);
    }
}

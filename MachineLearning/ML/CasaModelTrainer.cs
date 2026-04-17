using System;
using MachineLearning.Models;
using Microsoft.ML;

namespace MachineLearning.ML;

public class CasaModelTrainer
{
    private MLContext mLContext = new MLContext();
    private IDataView dados;
    public void CarregarDadosCSV(string path)
    {
        dados = mLContext.Data.LoadFromTextFile<CasaInputData>(
            path: path,
            hasHeader: true,
            separatorChar: ','
        );
    }
}

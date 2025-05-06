"""Module to build and compare models."""
import pandas as pd
import numpy as np 
import geopandas as gpd
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler,PolynomialFeatures, scale
from sklearn.model_selection import LeaveOneOut, train_test_split, cross_val_score, RepeatedKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.pipeline import Pipeline
from numpy import arange
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def prepare_data(gdf, cols_to_remove, crs = "EPSG:4326"):
    """
    This function removes unnecessary columns and normalizes remaining features by 
    area of census subdivision. 

    Parameters:
        df: gdf, assume contains features and target
        cols_to_remove: list
    """

    # A. Project to new CRS
    gdf_proj = gdf.to_crs(crs)

    # B. Remove Yukon
    gdf_no_yukon = gdf_proj[gdf_proj["Subdivision"]!="Yukon"].copy()
    subdivisions = gdf_no_yukon["Subdivision"]

    # C. Convert area unit from m to km
    gdf_no_yukon["area_km_sq"]=gdf_no_yukon.area/10**6
    areas = gdf_no_yukon["area_km_sq"]

    # D. Drop columns
    gdf_reduced = gdf_no_yukon.drop(columns=cols_to_remove) 

    # E. Find normalization. NOTE: This makes our response pop density, not pop count. 
    gdf_norm = gdf_reduced.apply(lambda row: row / row["area_km_sq"], axis=1)
    gdf_norm["population_density"] = gdf_norm["population"]

    # F. Remove area
    gdf_reduced = gdf_norm.drop(columns=["area_km_sq", "population"])
    

    return(gdf_reduced, areas, subdivisions)


# Function where I can build the regressors 

def compare_models(regressors, gdf, degrees=[1, 2]):
    """
    Compares different models using MAE and optionally includes polynomial features.

    Parameters:
        regressors: dict of regressors
        gdf: GeoDataFrame with features and target
        degrees: list of polynomial degrees to try (default=[1, 2])
    """
    data = []
   
    y = gdf["population_density"]
    X = gdf.drop(columns=["population_density"])
    loo = LeaveOneOut()

    for name, model in regressors.items():
        for degree in degrees:
            # Create a pipeline with optional polynomial features
            pipeline = Pipeline([
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),  # Add polynomial terms
                ("scaler", StandardScaler()),  # Ensure scaling after polynomial transformation
                ("regressor", model)
            ])

            # Track convergence and warnings
            converged = True
            warning_raised = None
            best_params = None
            try:
                if name in ['LinearRegression', 'RandomForest', 'GradientBoosting']:
                    # Use cross_val_predict for models that don't support built-in CV
                    y_pred = cross_val_predict(pipeline, X, y, cv=loo)
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", category=ConvergenceWarning)  # Treat ConvergenceWarning as an error
                        warnings.filterwarnings("error", category=UndefinedMetricWarning)  # Treat UndefinedMetricWarning as an error
                        warnings.filterwarnings("error", category=UserWarning)  # Treat UserWarning as an error
                        pipeline.fit(X, y)
                        y_pred = pipeline.predict(X)

                    # Extract best hyperparameters for models that support it
                    if hasattr(model, "alpha_"):  # For RidgeCV, LassoCV, ElasticNetCV
                        best_params = f"alpha={model.alpha_}"
                    if hasattr(model, "l1_ratio_"):  # For ElasticNetCV
                        best_params += f", l1_ratio={model.l1_ratio_}"

            except ConvergenceWarning:
                converged = False
                warning_raised = "ConvergenceWarning"
                y_pred = [0] * len(y)  # Set predictions to 0 if the model fails to converge
            except UndefinedMetricWarning:
                converged = False
                warning_raised = "UndefinedMetricWarning"
                y_pred = [0] * len(y)  # Handle undefined metrics gracefully
            except UserWarning:
                converged = False
                warning_raised = "UserWarning"
                y_pred = [0] * len(y)  # Handle user warnings gracefully

            # Calculate metrics
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            new_row = [name, degree, mae, mse, converged, warning_raised, best_params]
            data.append(new_row)
    
    # Create DataFrame with results
    df_to_return = pd.DataFrame(data, columns=["Regressor", "Polynomial Degree", "MAE", "MSE", "Converged", "Warning", "Best Params"]).sort_values(by="MAE", ascending=True)
    return df_to_return



if __name__ == "__main__":
    gdf = gpd.read_file("Data/osm_extractor_output")

    cols_to_remove = ["geo_point_2d",
                    "year",
                    "prov_code",
                    "prov_name_en",
                    "cd_code",
                    "cd_name_en",
                    "csd_code",
                    "csd_name_en",
                    "csd_area_code",
                    "csd_type",
                    "prov_name_fr",
                    "cd_name_fr",
                    "csd_name_fr",
                    "Subdivision",
                    "updated",
                    "geometry"]
    gdf_norm, areas, subdivisions = prepare_data(gdf, cols_to_remove, crs='epsg:3857')

        # Leave-One-Out CV
    loo = LeaveOneOut()

    regressors = {
        'LinearRegression': LinearRegression(),
        'RidgeCV': RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=loo),
        'LassoCV': LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=loo, max_iter=10000),
        'ElasticNetCV': ElasticNetCV(alphas=[0.001, 0.01, 0.1, 1, 10],
                                     l1_ratio=[0.1, 0.5, 0.9],
                                     cv=loo,
                                     max_iter=10000),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                      max_depth=3, random_state=42)
    }

    performance = compare_models(regressors, gdf_norm, degrees=[1, 2])
    print(performance)


        

    

    















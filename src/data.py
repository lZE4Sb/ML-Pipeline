import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
    TargetEncoder,
)

cfg = OmegaConf.load('config.yaml')

##### COLUMNS GROUPS #####

FILL_NONE_COLS = ['Alley', 'BsmtQual', 'BsmtCond', 
                  'BsmtExposure', 'BsmtFinType1', 
                  'BsmtFinType2', 'FireplaceQu', 
                  'GarageType', 'GarageFinish', 
                  'GarageQual', 'GarageCond', 
                  'PoolQC', 'Fence', 'MiscFeature']

FILL_MEDIAN_COLS = ['LotFrontage', 'LotArea', 'YearBuilt', 
                    'YearRemodAdd', 'GarageYrBlt', 'MasVnrArea', 
                    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
                    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
                    'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 
                    'BsmtHalfBath', 'FullBath', 'HalfBath', 
                    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 
                    'Fireplaces', 'GarageCars', 'GarageArea', 
                    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
                    '3SsnPorch', 'ScreenPorch','PoolArea', 
                    'MiscVal', 'MoSold', 'YrSold']

FILL_MOST_COMMON_COLS = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 
                         'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                         'LandSlope', 'Neighborhood', 'Condition1', 
                         'Condition2', 'BldgType', 'HouseStyle', 
                         'OverallQual', 'OverallCond', 
                         'RoofStyle', 'RoofMatl', 
                         'Exterior1st', 'Exterior2nd', 'MasVnrType', 
                         'ExterQual', 'ExterCond', 'Foundation', 
                         'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                         'BsmtFinType1', 'BsmtFinType2', 'Heating', 
                         'HeatingQC', 'CentralAir', 'Electrical', 
                         'KitchenQual', 'Functional', 'FireplaceQu', 
                         'GarageType', 'GarageFinish', 'GarageQual', 
                         'GarageCond', 'PavedDrive', 'PoolQC', 
                         'Fence', 'MiscFeature', 'SaleType', 
                         'SaleCondition']

NOMINATIVE_FEATURES = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 
                       'LandContour', 'LotConfig', 'Neighborhood', 
                       'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
                       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
                       'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 
                       'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']

NUM_COLS = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd',
            'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
            'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
            'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
            'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
            'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 
            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
            'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

##### ORDINAL COLUMNS AND CATEGORIES #####

FIVE_LEVELS = ['Po', 'Fa', 'T0A', 'Gd', 'Ex']
TEN_LEVELS = list(range(10))

ORDERED_LEVELS = {
    'OverallQual': TEN_LEVELS,
    'OverallCond': TEN_LEVELS,
    'ExterQual': FIVE_LEVELS,
    'ExterCond': FIVE_LEVELS,
    'BsmtQual': ['None', 'Po', 'Fa', 'T0A', 'Gd', 'Ex'],
    'BsmtCond': ['None', 'Po', 'Fa', 'T0A', 'Gd', 'Ex'],
    'HeatingQC': FIVE_LEVELS,
    'KitchenQual': FIVE_LEVELS,
    'FireplaceQu': ['None', 'Po', 'Fa', 'T0A', 'Gd', 'Ex'],
    'GarageQual': ['None', 'Po', 'Fa', 'T0A', 'Gd', 'Ex'],
    'GarageCond': ['None', 'Po', 'Fa', 'T0A', 'Gd', 'Ex'],
    'PoolQC': ['None', 'Po', 'Fa', 'T0A', 'Gd', 'Ex'],
    'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],
    'LandSlope': ['Sev', 'Mod', 'Gtl'],
    'BsmtExposure': ['None', 'No', 'Mn', 'Av', 'Gd'],
    'BsmtFinType1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'BsmtFinType2': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'Functional': ['Sal', 'Sev', 'Maj1', 'Maj2', 'Mod', 'Min2', 'Min1', 'Typ'],
    'GarageFinish': ['None', 'Unf', 'RFn', 'Fin'],
    'PavedDrive': ['N', 'P', 'Y'],
    'Utilities': ['NoSeWa', 'NoSewr', 'AllPub'],
    'CentralAir': ['N', 'Y'],
    'Electrical': ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'],
    'Fence': ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
}

ORDINAL_COLS = list(ORDERED_LEVELS.keys())
CATEGORIES = [ORDERED_LEVELS[col] for col in ORDINAL_COLS]

##### SKEWED COLS #####

QUANTILE_COLS = ['LotArea', 'MasVnrArea', 'BsmtFinSF2', 
                 'LowQualFinSF', 'BsmtHalfBath', 'KitchenAbvGr', 
                 'GarageYrBlt', 'OpenPorchSF', 'EnclosedPorch', 
                 '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

YEO_JOHNSON_COLS = ['BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 
                    '2ndFlrSF', 'BsmtFullBath', 'HalfBath', 
                    'Fireplaces', 'WoodDeckSF']

BOX_COX_COLS = ['LotFrontage', 'YearBuilt', '1stFlrSF', 
                'GrLivArea', 'TotRmsAbvGrd']


##### DATA PREPARATION FUNCTIONS #####

def load_data(cfg):
    '''Load the dataset from the CSV file.'''
    df = pd.read_csv(cfg.paths.train_csv, index_col=cfg.data.id_col)
    if cfg.general.debug: #takes first 10 percent of dataset
        part = round(len(df) / 10)
        df = df.head(part)
    return df

def stateless_preprocessing(df):
    '''Fill missing values that means lack of house option with 'None' or 0.0.'''
    df[FILL_NONE_COLS] = df[FILL_NONE_COLS].fillna('None')
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    return df

def make_test_split(cfg, df):
    '''Split the dataset into training and testing sets if test size > 0'''
    if cfg.data.test_size > 0:
        df, test = train_test_split(df, 
                                    test_size=cfg.data.test_size,
                                    random_state=cfg.general.seed)
    else:
        test = None
    return df, test

def make_pipeline(cfg, df):    
    imputer_transformer = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), FILL_MEDIAN_COLS), 
            ('cat', SimpleImputer(strategy='most_frequent'), FILL_MOST_COMMON_COLS) 
        ], verbose_feature_names_out=False, 
            remainder='passthrough').set_output(transform='pandas')

    ordinal_encoder = OrdinalEncoder(
        categories = CATEGORIES,
        dtype = np.int64, #cos CatBoost treat int64 as ordinal categories
        handle_unknown = 'use_encoded_value',
        unknown_value = -1,
        encoded_missing_value = -2
    )
    ordinal_transformer = ColumnTransformer(
        transformers=[
            ('ord', ordinal_encoder, ORDINAL_COLS)
        ],
        verbose_feature_names_out=False,
        remainder='passthrough'
    ).set_output(transform='pandas')

    target_encoder = TargetEncoder(
        target_type = 'continuous',
        smooth = 20.0,
        cv = 5,
        shuffle = True,
        random_state = cfg.general.seed,
        )

    target_transformer = ColumnTransformer(
        transformers=[
            ('cat', target_encoder, NOMINATIVE_FEATURES)],
            verbose_feature_names_out=False, 
            remainder='passthrough').set_output(transform='pandas')

    quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=cfg.general.seed)
    yeo_johnson_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    box_cox_transformer = PowerTransformer(method='box-cox', standardize=True)

    skewness_transformer = ColumnTransformer(
        transformers=[
                ('quant', quantile_transformer, QUANTILE_COLS),
                ('yeo', yeo_johnson_transformer, YEO_JOHNSON_COLS),
                ('box', box_cox_transformer, BOX_COX_COLS)],
            verbose_feature_names_out=False, 
            remainder='passthrough').set_output(transform='pandas')

    standard_transformer = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), df.drop(cfg.data.target_col, axis=1).columns) # i'm gonna scale all, so no need to choose
        ],
        verbose_feature_names_out=False,
        remainder='passthrough' 
    ).set_output(transform='pandas')

    categorical_encoder = OrdinalEncoder(
        categories = 'auto',
        handle_unknown='use_encoded_value', # this encoder for XGB only, cos he cant handle unknown cat values
        unknown_value=-1,
        encoded_missing_value=-2
    )

    categorical_transformer = ColumnTransformer(
        transformers=[
            ('cat', categorical_encoder, NOMINATIVE_FEATURES)],
            verbose_feature_names_out=False, 
            remainder='passthrough').set_output(transform='pandas') 

    if cfg.data.pipeline == 'linear':
        pipeline = Pipeline(steps=[
            ('imputer', imputer_transformer),
            ('ordinal', ordinal_transformer),
            ('target', target_transformer),
            ('skewness', skewness_transformer),
            ('scaler', standard_transformer)
        ])
    elif cfg.data.pipeline == 'boosting':
        if cfg.model_class == 'xgb':
            pipeline = Pipeline(steps=[
                ('ordinal', ordinal_transformer),
                ('cat', categorical_transformer)
            ]) # for XGB only, cos he cant handle unknown cat values
        else:
            pipeline = Pipeline(steps=[
                ('ordinal', ordinal_transformer),
            ])

    return pipeline

def create_splits(cfg, df, col_name = 'fold'
) -> pd.DataFrame: 
    '''
    Creates train/validation or k-fold split column for the given DataFrame.
    Uses bin stratification for continuous target if specified.
    '''
    df = df.copy()

    target_col = cfg.data.target_col
    split_strategy = cfg.data.split_strategy
    n_splits = cfg.data.n_splits
    val_size = cfg.data.val_size
    stratify = cfg.data.stratify
    n_bins_stratify = cfg.data.n_splits
    random_state = cfg.general.seed
    
    # 1. Stratification
    stratify_target = None
    if stratify:
        if target_col is None or target_col not in df.columns:
            raise ValueError('Need to specify `target_col`')
        
        # Bin continuous target into quantiles for stratification if it's numeric and has many unique values
        if np.issubdtype(df[target_col].dtype, np.number) and df[target_col].nunique() > 20:
            stratify_target = pd.qcut(df[target_col], q=n_bins_stratify, labels=False, duplicates='drop')
        else:
            stratify_target = df[target_col]

    # 2. Train / Validation
    if split_strategy == 'train_val':
        df[col_name] = 'train'
        
        if stratify:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
            _, val_idx = next(splitter.split(df, stratify_target))
        else:
            splitter = ShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
            _, val_idx = next(splitter.split(df))
            
        df.iloc[val_idx, df.columns.get_loc(col_name)] = 'val'

    # 3. K-Fold Cross-Validation
    elif split_strategy == 'kfold':
        df[col_name] = -1
        
        if stratify:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_generator = skf.split(df, stratify_target)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_generator = kf.split(df)
            
        for fold, (_, val_idx) in enumerate(split_generator):
            df.iloc[val_idx, df.columns.get_loc(col_name)] = fold

    else:
        raise ValueError(f"Unknown strategy '{split_strategy}'. Available strategies: 'kfold', 'train_val'")

    return df

### IN-CV CYCLE FUNCTION ###

def prepare_data_by_folds(cfg, df, pipeline, fold=0):
    '''
    Prepares training and validation data through
    preprocessing pipeline for a specific fold.
    
    Args:
        cfg: config dict
        df: DataFrame containing the data with a 'fold' column
        pipeline: Preprocessing sklearn pipeline to transform features
        fold: Current fold number (int)
        '''
    target_col = cfg.data.target_col

    X_train = df[df['fold'] != fold].drop(['fold', target_col], axis=1)
    y_train = df[df['fold'] != fold][target_col]
    X_val = df[df['fold'] == fold].drop(['fold', target_col], axis=1)
    y_val = df[df['fold'] == fold][target_col]

    X_train = pipeline.fit_transform(X_train, y_train)
    X_val = pipeline.transform(X_val)

    return X_train, y_train, X_val, y_val



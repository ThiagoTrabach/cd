
from sqlalchemy import create_engine
import numpy as np
import pandas as pd


# In[3]:


engine = create_engine()
e = engine.connect()


print('inicio processo')

phones = pd.read_sql('select distinct users.base_user_id, leads.phone_id as contact_id, \'whatsapp\' as channel  from gambit.users join gambit.leads on leads.global_user_id = users.global_user_id where phone_id is not null',e)

phones_clean = phones.groupby('contact_id').first().reset_index()

emails = pd.read_sql('select distinct users.base_user_id, email_address_id as contact_id, \'email\' as channel from gambit.users join gambit.leads on leads.global_user_id = users.global_user_id where email_address_id is not null', e)

emails_clean = emails.groupby('contact_id').first().reset_index()


del phones
del emails


users = pd.concat([phones_clean,emails_clean])

users_model = users.groupby('base_user_id').first().reset_index()


print('Total de usuarios com email/telefone cadastrados:',len(users_model))


del users


print('-----------------------------------------------')
print('Total de usuarios:','    ',len(users_model))


#users_random_full = pd.concat([users_model, users_model,users_model,users_model]).reset_index(drop=True)

ofertas_ativas = pd.read_sql('select distinct offers.id as id_offer, cities.name as city, courses.canonical_course_id, offers.offered_price, rank() OVER (PARTITION BY cities.name,canonical_course_id  ORDER BY city,canonical_course_id,offers.offered_price) from offers join courses on courses.id = offers.course_id join campuses on campuses.id = courses.campus_id join cities on cities.name = campuses.city and cities.state = campuses.state where offers.enabled = True and offers.visible=True and offers.restricted=False order by courses.canonical_course_id, offers.offered_price, cities.name',e)

ofertas_ativas = ofertas_ativas[(ofertas_ativas['rank']==1)].reset_index(drop=True)

#ofertas_aleatorias = ofertas_ativas.sample(n=(len(users_random_full)), replace=True).reset_index(drop=True)

#output_data_test = pd.concat([users_random_full[['base_user_id','contact_id','channel']],ofertas_aleatorias['id_offer']], axis=1)

#output_data_test['model_version'] = '-1.0'

#print('fim modelo random')

search = pd.read_sql('select distinct base_user_id, canonical_course_id, city, first_value(interests.id) over (partition by interests.base_user_id order by interests.created_at desc) id, rank() OVER (PARTITION BY interests.base_user_id ORDER BY canonical_course_id,city) from interests  inner join canonical_courses on canonical_courses.id = interests.canonical_course_id where canonical_course_id is not null and interests.course_level is not null and base_user_id is not null',e)

search_clean = search[search['rank']==1]

rec_search_off = pd.merge(users_model, search_clean, on='base_user_id',how='inner')

search_clean_als = search[search['rank']<=5]

search_clean_als_off = pd.merge(users_model, search_clean_als, on='base_user_id',how='inner')




# # Modelo V1

import scipy.sparse as sparse
import pickle
import implicit
import itertools
import copy

print('inicio modelo v1')

df = search_clean_als_off[['base_user_id','canonical_course_id']].reset_index(drop=True)
del search
del search_clean
del rec_search_off
del search_clean_als
del search_clean_als_off

df['base_user_id'] = df.base_user_id.astype('int')
df['base_user_id'] = df.base_user_id.astype('object')
df['canonical_course_id'] = df.canonical_course_id.astype('object')

print('Duplicated rows: ' + str(df.duplicated().sum()))
df.drop_duplicates(inplace=True)

n_users = df.base_user_id.unique().shape[0]
n_items = df.canonical_course_id.unique().shape[0]
print ('Number of users = ' + str(n_users) + ' | Number of courses = ' + str(n_items))

n_users = df.base_user_id.unique().shape[0]
n_items = df.canonical_course_id.unique().shape[0]

print('Number of users: {}'.format(n_users))
print('Number of models: {}'.format(n_items))
print('Sparsity: {:4.3f}%'.format(float(df.shape[0]) / float(n_users*n_items) * 100))


# Create mappings
canonical_course_id_to_idx = {}
idx_to_canonical_course_id = {}
for (idx, canonical_course_id) in enumerate(df.canonical_course_id.unique().tolist()):
    canonical_course_id_to_idx[canonical_course_id] = idx
    idx_to_canonical_course_id[idx] = canonical_course_id

base_user_id_to_idx = {}
idx_to_base_user_id = {}
for (idx, base_user_id) in enumerate(df.base_user_id.unique().tolist()):
    base_user_id_to_idx[base_user_id] = idx
    idx_to_base_user_id[idx] = base_user_id


def map_ids(row, mapper):
    return mapper[row]


I = df.base_user_id.apply(map_ids, args=[base_user_id_to_idx]).as_matrix()
J = df.canonical_course_id.apply(map_ids, args=[canonical_course_id_to_idx]).as_matrix()
V = np.ones(I.shape[0])
likes = sparse.coo_matrix((V, (I, J)), dtype=np.float64)
likes = likes.tocsr()


def train_test_split(ratings, split_count, fraction=None):
    train = ratings.copy().tocoo()
    test = sparse.lil_matrix(train.shape)
    if fraction:
        try:
            user_index = np.random.choice(
                np.where(np.bincount(train.row) >= split_count * 2)[0],
                replace=True,
                size=np.int32(np.floor(fraction * train.shape[0]))
            ).tolist()
        except:
            print(('Not enough users with > {} '
                  'interactions for fraction of {}'))
            raise
    else:
        user_index = range(train.shape[0])
    train = train.tolil()
    for user in user_index:
        test_ratings = np.random.choice(ratings.getrow(user).indices,
                                        size=split_count,
                                        replace=True)
        train[user, test_ratings] = 0.
        # These are just 1.0 right now
        test[user, test_ratings] = ratings[user, test_ratings]
    assert(train.multiply(test).nnz == 0)
    return train.tocsr(), test.tocsr(), user_index


# In[123]:

print('train e test')

train, test, user_index = train_test_split(likes, 4, fraction=0)


# In[124]:


from sklearn.metrics import mean_squared_error
def calculate_mse(model, ratings, user_index=None):
    preds = model.predict_for_customers()
    if user_index:
        return mean_squared_error(ratings[user_index, :].toarray().ravel(),
                                  preds[user_index, :].ravel())
    return mean_squared_error(ratings.toarray().ravel(),
                              preds.ravel())


# In[125]:


def precision_at_k(model, ratings, k=4, user_index=None):
    if not user_index:
        user_index = range(ratings.shape[0])
    ratings = ratings.tocsr()
    precisions = []
    # Note: line below may become infeasible for large datasets.
    predictions = model.predict_for_customers()
#    user_items = train.T.tocsr()
#    predictions = model.recommend(user_index, user_items)
    for user in user_index:
        # In case of large dataset, compute predictions row-by-row like below
        # predictions = np.array([model.predict(row, i) for i in xrange(ratings.shape[1])])
        top_k = np.argsort(-predictions[user, :])[:k]
        labels = ratings.getrow(user).indices
        precision = float(len(set(top_k) & set(labels))) / float(k)
        precisions.append(precision)
    return np.mean(precisions)


# In[126]:


def print_log(row, header=False, spacing=12):
    top = ''
    middle = ''
    bottom = ''
    for r in row:
        top += '+{}'.format('-'*spacing)
        if isinstance(r, str):
            middle += '| {0:^{1}} '.format(r, spacing-2)
        elif isinstance(r, int):
            middle += '| {0:^{1}} '.format(r, spacing-2)
        elif isinstance(r, float):
            middle += '| {0:^{1}.5f} '.format(r, spacing-2)
        bottom += '+{}'.format('='*spacing)
    top += '+'
    middle += '|'
    bottom += '+'
    if header:
        print(top)
        print(middle)
        print(bottom)
    else:
        print(middle)
        print(top)


# In[127]:


def learning_curve(model, train, test, epochs, k=4, user_index=None):
    if not user_index:
        user_index = range(train.shape[0])
    prev_epoch = 0
    train_precision = []
    train_mse = []
    test_precision = []
    test_mse = []
    headers = ['epochs', 'p@k train', 'p@k test',
               'mse train', 'mse test']
    print_log(headers, header=True)
    for epoch in epochs:
        model.iterations = epoch - prev_epoch
        if not hasattr(model, 'user_vectors'):
            model.fit(train)
        else:
            model.fit_partial(train)
        train_mse.append(calculate_mse(model, train, user_index))
        train_precision.append(precision_at_k(model, train, k, user_index))
        test_mse.append(calculate_mse(model, test, user_index))
        test_precision.append(precision_at_k(model, test, k, user_index))
        row = [epoch, train_precision[-1], test_precision[-1],
               train_mse[-1], test_mse[-1]]
        print_log(row)
        prev_epoch = epoch
    return model, train_precision, train_mse, test_precision, test_mse


# In[128]:


def grid_search_learning_curve(base_model, train, test, param_grid, user_index=None, patk=4, epochs=range(2, 40, 2)):
    curves = []
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        this_model = copy.deepcopy(base_model)
        print_line = []
        for k, v in params.items():
            setattr(this_model, k, v)
            print_line.append((k, v))
        print(' | '.join('{}: {}'.format(k, v) for (k, v) in print_line))
        _, train_patk, train_mse, test_patk, test_mse = learning_curve(this_model, train, test,
                                                                epochs, k=patk, user_index=user_index)
        curves.append({'params': params,
                       'patk': {'train': train_patk, 'test': test_patk},
                       'mse': {'train': train_mse, 'test': test_mse}})
    return curves


# In[129]:


param_grid = {'num_factors': [10], 'regularization': [1e-5], 'alpha': [10]}


# In[130]:


base_model = implicit.ALS()


# In[131]:
print('inicio treino')

curves = grid_search_learning_curve(base_model, train, test, param_grid, user_index=user_index,patk=4)


best_curves = sorted(curves, key=lambda x: max(x['patk']['test']), reverse=True)


print(best_curves[0]['params'])
max_score = max(best_curves[0]['patk']['test'])
print(max_score)
iterations = range(2, 40, 2)[best_curves[0]['patk']['test'].index(max_score)]
print('Epoch: {}'.format(iterations))


all_test_patks = [x['patk']['test'] for x in best_curves]


params = best_curves[0]['params']
params['iterations'] = range(2, 40, 2)[best_curves[0]['patk']['test'].index(max_score)]
bestALS = implicit.ALS(**params)

bestALS.fit(likes)

user_items = bestALS.predict_for_customers()
user_item_dict = {}

for idx, user in enumerate(user_items):
    items = list(np.argsort(user))[:4]
    user_item_dict[idx] = items

users_uniques1 = pd.DataFrame(df.base_user_id.unique(), columns=['base_user_id'])
users_uniques2 = pd.DataFrame(df.base_user_id.unique(), columns=['base_user_id'])
users_uniques3 = pd.DataFrame(df.base_user_id.unique(), columns=['base_user_id'])
users_uniques4 = pd.DataFrame(df.base_user_id.unique(), columns=['base_user_id'])


courses =[]
for i in user_item_dict:
    courses.append(user_item_dict[i])


# In[139]:



search = pd.read_sql('select distinct base_user_id, canonical_course_id, city, first_value(interests.id) over (partition by interests.base_user_id order by interests.created_at desc) id, rank() OVER (PARTITION BY interests.base_user_id ORDER BY canonical_course_id,city) from interests  inner join canonical_courses on canonical_courses.id = interests.canonical_course_id where canonical_course_id is not null and interests.course_level is not null and base_user_id is not null',e)

search_clean = search[search['rank']==1]


search_clean_als = search[search['rank']<=5]

del search

rec_search_off = pd.merge(users_model, search_clean, on='base_user_id',how='inner')

del search_clean

search_clean_als_off = pd.merge(users_model, search_clean_als, on='base_user_id',how='inner')

del search_clean_als

courses =[]
for i in user_item_dict:
    courses.append(user_item_dict[i])

users_uniques1 = pd.DataFrame(df.base_user_id.unique(), columns=['base_user_id'])
users_uniques2 = pd.DataFrame(df.base_user_id.unique(), columns=['base_user_id'])
users_uniques3 = pd.DataFrame(df.base_user_id.unique(), columns=['base_user_id'])
users_uniques4 = pd.DataFrame(df.base_user_id.unique(), columns=['base_user_id'])


users_uniques1['canonical_course_id'] = [row[0] for row in courses]
users_uniques2['canonical_course_id'] = [row[1] for row in courses]
users_uniques3['canonical_course_id'] = [row[2] for row in courses]

users_uniques = pd.concat([rec_search_off[['base_user_id','canonical_course_id']],users_uniques1,users_uniques2,users_uniques3])


users_uniques['base_user_id'] = users_uniques['base_user_id'].astype('int')


del users_uniques1
del users_uniques2
del users_uniques3
del users_uniques4


modelo_v1 = pd.merge(search_clean_als_off[['base_user_id','contact_id','city', 'channel']], users_uniques[['base_user_id', 'canonical_course_id']], on=['base_user_id'],how='inner').drop_duplicates()


rec_modelo_v1 = pd.merge(modelo_v1[['base_user_id','contact_id','city', 'canonical_course_id','channel']], ofertas_ativas[['city', 'canonical_course_id','id_offer']], on=['city', 'canonical_course_id'],how='inner').drop_duplicates()

del modelo_v1
del ofertas_ativas

rec_modelo_v1['model_version'] = '2.0'


print('consolidacao modelos')
#output = rec_modelo_v1


del likes

#output.drop_duplicates(inplace =True)

# calculo prob

rec_buy = pd.read_sql('select  distinct users.global_user_id, orders.base_user_id, canonical_course_id as buy from orders join line_items on orders.id = line_items.order_id join offers on line_items.offer_id = offers.id join courses on offers.course_id = courses.id join canonical_courses on canonical_courses.id = courses.canonical_course_id join gambit.users on users.base_user_id = orders.base_user_id  where  orders.checkout_step = \'paid\' and not dirty', e)

rec_search = pd.read_sql('select distinct base_user_id, canonical_course_id as searched, first_value(interests.id) over (partition by interests.base_user_id order by interests.created_at desc) id from interests join canonical_courses on canonical_courses.id = interests.canonical_course_id where canonical_course_id is not null and interests.course_level is not null',e)

juncao = pd.merge(rec_buy, rec_search, on='base_user_id',how='inner')

del rec_buy
del rec_search

taxa = juncao.groupby(['searched'])['base_user_id'].count().reset_index(name='qtd_full')
baseline = juncao.groupby(['searched','buy'])['base_user_id'].count().reset_index(name='qtd')
baseline.sort_values(by=['searched', 'qtd'], ascending=[True, False], inplace=True)
baseline.reset_index(inplace=True, drop=True)
baseline['canonical_course_id'] = baseline.searched

df_taxa = pd.merge(baseline, taxa, on=['searched'],how='left').drop_duplicates()

df_taxa['prob'] = df_taxa['qtd']/df_taxa['qtd_full']

model_p = pd.merge(rec_modelo_v1, rec_search_off[['contact_id', 'channel','canonical_course_id']].rename(columns={"canonical_course_id": "searched"}), on=['contact_id', 'channel'],how='inner').drop_duplicates()
output = pd.merge(model_p, df_taxa[['searched','buy','prob']].rename(columns = {"buy":"canonical_course_id"}), on=['searched', 'canonical_course_id'],how='left').drop_duplicates()

del rec_search_off

output['prob']= output.prob.replace(np.nan, 0.01)

output.drop_duplicates(inplace =True)


output['last_update_recommendation'] = pd.Timestamp.today()



output=output[output.id_offer.notnull()].reset_index(drop=True)

engine_2 = create_engine()
e2 = engine_2.connect()


# In[156]:

print('copia base')

output.to_sql('output_api_t',e2, if_exists='append')

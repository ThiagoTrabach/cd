# coding: utf-8

# In[2]:


from sqlalchemy import create_engine
import numpy as np
import pandas as pd


# In[3]:


engine = create_engine('postgresql://')
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




ofertas_ativas = pd.read_sql('select distinct offers.id as id_offer, cities.name as city, courses.canonical_course_id, offers.offered_price, rank() OVER (PARTITION BY cities.name,canonical_course_id  ORDER BY city,canonical_course_id,offers.offered_price) from offers join courses on courses.id = offers.course_id join campuses on campuses.id = courses.campus_id join cities on cities.name = campuses.city and cities.state = campuses.state where offers.enabled = True and offers.visible=True and offers.restricted=False order by courses.canonical_course_id, offers.offered_price, cities.name',e)

ofertas_ativas = ofertas_ativas[(ofertas_ativas['rank']==1)].reset_index(drop=True)


search = pd.read_sql('select distinct base_user_id, canonical_course_id, city, first_value(interests.id) over (partition by interests.base_user_id order by interests.created_at desc) id, rank() OVER (PARTITION BY interests.base_user_id ORDER BY canonical_course_id,city) from interests  inner join canonical_courses on canonical_courses.id = interests.canonical_course_id where canonical_course_id is not null and interests.course_level is not null and base_user_id is not null',e)



search_clean = search[search['rank']==1]


rec_search_off = pd.merge(users_model, search_clean, on='base_user_id',how='inner')





rec_buy = pd.read_sql('select  distinct users.global_user_id, orders.base_user_id, canonical_course_id as buy from orders join line_items on orders.id = line_items.order_id join offers on line_items.offer_id = offers.id join courses on offers.course_id = courses.id join canonical_courses on canonical_courses.id = courses.canonical_course_id join gambit.users on users.base_user_id = orders.base_user_id  where  orders.checkout_step = \'paid\' and not dirty', e)

rec_search = pd.read_sql('select distinct base_user_id, canonical_course_id as searched, first_value(interests.id) over (partition by interests.base_user_id order by interests.created_at desc) id from interests join canonical_courses on canonical_courses.id = interests.canonical_course_id where canonical_course_id is not null and interests.course_level is not null',e)

juncao = pd.merge(rec_buy, rec_search, on='base_user_id',how='inner')


del rec_buy
del rec_search

print('inicio baseline')
baseline = juncao.groupby(['searched','buy'])['base_user_id'].count().reset_index(name='qtd')

baseline.sort_values(by=['searched', 'qtd'], ascending=[True, False], inplace=True)

baseline.reset_index(inplace=True, drop=True)


baseline['canonical_course_id'] = baseline.searched

baseline['rank'] = baseline.groupby(['searched'])['qtd'].rank(ascending=False)

model_baseline = baseline[baseline['rank']<=4]


len(model_baseline)

cliente_rec = pd.merge(rec_search_off, model_baseline, on='canonical_course_id',how='left')

cliente_rec.head()


ofertas_ativas['buy'] = ofertas_ativas.canonical_course_id


rec_baseline = pd.merge(cliente_rec[['base_user_id','contact_id','city', 'buy','channel']], ofertas_ativas[['city', 'buy','id_offer']], on=['city', 'buy'],how='left').drop_duplicates()


rec_baseline.drop_duplicates(inplace=True)


rec_baseline['model_version'] = '0.0'
rec_baseline['canonical_course_id'] = rec_baseline.buy

print('fim baseline')




#rec_baseline = rec_baseline[['channel','model_version','contact_id','base_user_id','id_offer','canonical_course_id']]


# calculo prob

taxa = juncao.groupby(['searched'])['base_user_id'].count().reset_index(name='qtd_full')

df_taxa = pd.merge(baseline, taxa, on=['searched'],how='left').drop_duplicates()

df_taxa['prob'] = df_taxa['qtd']/df_taxa['qtd_full']

model_p = pd.merge(rec_baseline, rec_search_off[['contact_id', 'channel','canonical_course_id']].rename(columns={"canonical_course_id": "searched"}), on=['contact_id', 'channel'],how='inner').drop_duplicates()
output = pd.merge(model_p, df_taxa[['searched','buy','prob']].rename(columns = {"buy":"canonical_course_id"}), on=['searched', 'canonical_course_id'],how='left').drop_duplicates()

output['prob']= output.prob.replace(np.nan, 0.01)


output['last_update_recommendation'] = pd.Timestamp.today()

output.drop_duplicates(inplace =True)

output=output[output.id_offer.notnull()].reset_index(drop=True)

engine_2 = create_engine('')
e2 = engine_2.connect()


# In[156]:

print('copia base')

output[['channel','model_version','contact_id','base_user_id','id_offer', 'prob', 'last_update_recommendation']].to_sql('output_api_t',e2, if_exists='append')

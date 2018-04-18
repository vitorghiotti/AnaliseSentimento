import sys
from pony import orm

non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
db = orm.Database()

class Reacao(db.Entity):
    id_cliente = orm.Required(int)
    id_fonte = orm.Required(int)
    reacao = orm.Required(str)
    sentimento = orm.Required(int)
    datatime_reacao = orm.Required(float)
    identificador_fonte = orm.Required(str)

db.bind('sqlite', 'tweets.sqlite', create_db=True)
db.generate_mapping(create_tables=True)


@orm.db_session
def add_reacao_tweet(id_cliente,id_fonte, reacao, sentimento, datatime_reacao, identificador_fonte):
    text = reacao.text.translate(non_bmp_map)
    reacao = Reacao(id_cliente=id_cliente, id_fonte=id_fonte, reacao=text, sentimento=sentimento,datatime_reacao=datatime_reacao,identificador_fonte=identificador_fonte)
    orm.commit()
    
@orm.db_session
def add_reacao_face(id_cliente,id_fonte, reacao, sentimento, datatime_reacao, identificador_fonte):
    text = reacao.translate(non_bmp_map)
    reacao = Reacao(id_cliente=id_cliente, id_fonte=id_fonte, reacao=text, sentimento=sentimento,datatime_reacao=datatime_reacao,identificador_fonte=identificador_fonte)
    orm.commit()
    
@orm.db_session
def add_reacao_insta(id_cliente,id_fonte, reacao, sentimento, datatime_reacao, identificador_fonte):
    text = reacao.translate(non_bmp_map)
    reacao = Reacao(id_cliente=id_cliente, id_fonte=id_fonte, reacao=text, sentimento=sentimento,datatime_reacao=datatime_reacao,identificador_fonte=identificador_fonte)
    orm.commit()
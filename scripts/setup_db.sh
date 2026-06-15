#!/usr/bin/env bash
# Создаёт роль и БД для backend. Запуск:
#   sudo bash backend/scripts/setup_db.sh
set -euo pipefail

DB_NAME="${DB_NAME:-mltrading}"
DB_USER="${DB_USER:-mltrading}"
DB_PASS="${DB_PASS:-mltrading}"

# Роль (идемпотентно через DO-блок)
sudo -u postgres psql -v ON_ERROR_STOP=1 <<SQL
DO \$\$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${DB_USER}') THEN
      CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASS}';
   END IF;
END
\$\$;
SQL

# БД (CREATE DATABASE нельзя внутри DO/транзакции — создаём через createdb)
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
  sudo -u postgres createdb -O "${DB_USER}" "${DB_NAME}"
fi

echo "OK: роль ${DB_USER} и БД ${DB_NAME} готовы."

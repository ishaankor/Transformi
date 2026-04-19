/**
 * Discord Bot for Cloudflare Workers
 * Simplified version without heavy ML dependencies
 */

import { verifyKey } from 'discord-interactions';

// Discord Application Configuration
const DISCORD_TOKEN = 'YOUR_DISCORD_BOT_TOKEN';
const DISCORD_PUBLIC_KEY = 'YOUR_DISCORD_PUBLIC_KEY';
const DISCORD_APPLICATION_ID = 'YOUR_DISCORD_APPLICATION_ID';

// Discord API endpoints
const DISCORD_API_URL = 'https://discord.com/api/v10';

// Interaction types
const InteractionType = {
  PING: 1,
  APPLICATION_COMMAND: 2,
  MESSAGE_COMPONENT: 3,
  APPLICATION_COMMAND_AUTOCOMPLETE: 4,
  MODAL_SUBMIT: 5
};

// Response types
const InteractionResponseType = {
  PONG: 1,
  CHANNEL_MESSAGE_WITH_SOURCE: 4,
  DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE: 5,
  DEFERRED_UPDATE_MESSAGE: 6,
  UPDATE_MESSAGE: 7,
  APPLICATION_COMMAND_AUTOCOMPLETE_RESULT: 8,
  MODAL: 9
};

export default {
  async fetch(request, env) {
    // Handle CORS preflight requests
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        },
      });
    }

    // Only handle POST requests
    if (request.method !== 'POST') {
      return new Response('Method not allowed', { status: 405 });
    }

    // Verify Discord signature
    const signature = request.headers.get('X-Signature-Ed25519');
    const timestamp = request.headers.get('X-Signature-Timestamp');
    const body = await request.text();

    const isValidRequest = verifyKey(body, signature, timestamp, env.DISCORD_PUBLIC_KEY || DISCORD_PUBLIC_KEY);
    if (!isValidRequest) {
      return new Response('Invalid request signature', { status: 401 });
    }

    const interaction = JSON.parse(body);

    // Handle ping
    if (interaction.type === InteractionType.PING) {
      return new Response(JSON.stringify({ type: InteractionResponseType.PONG }), {
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // Handle application commands
    if (interaction.type === InteractionType.APPLICATION_COMMAND) {
      return await handleCommand(interaction, env);
    }

    // Handle message components (buttons, select menus)
    if (interaction.type === InteractionType.MESSAGE_COMPONENT) {
      return await handleComponent(interaction, env);
    }

    return new Response('Unknown interaction type', { status: 400 });
  },
};

async function handleCommand(interaction, env) {
  const { name } = interaction.data;

  switch (name) {
    case 'create_neural_network':
      return await handleNeuralNetworkCommand(interaction, env);
    case 'graph_linear_regression':
      return await handleLinearRegressionCommand(interaction, env);
    case 'ping':
      return jsonResponse({
        type: InteractionResponseType.CHANNEL_MESSAGE_WITH_SOURCE,
        data: {
          content: '🏓 Pong! Bot is running on Cloudflare Workers!',
          flags: 64 // Ephemeral
        }
      });
    default:
      return jsonResponse({
        type: InteractionResponseType.CHANNEL_MESSAGE_WITH_SOURCE,
        data: {
          content: 'Unknown command!',
          flags: 64
        }
      });
  }
}

async function handleNeuralNetworkCommand(interaction, env) {
  // Since we can't run actual ML training in Workers, we'll simulate it
  return jsonResponse({
    type: InteractionResponseType.CHANNEL_MESSAGE_WITH_SOURCE,
    data: {
      embeds: [{
        title: '🧠 Neural Network Simulation',
        description: 'Due to Cloudflare Workers limitations, actual ML training is not supported. However, here\'s what would happen:',
        fields: [
          {
            name: '📊 Simulated Training',
            value: 'MNIST dataset (1000 samples)\nAccuracy: ~85%\nLoss: ~0.3',
            inline: true
          },
          {
            name: '🏗️ Architecture',
            value: 'Input(784) → Dense(32) → Output(10)',
            inline: true
          },
          {
            name: '⚡ Alternative Solutions',
            value: 'Consider using:\n• Google Colab for training\n• TensorFlow.js for browser ML\n• External ML APIs',
            inline: false
          }
        ],
        color: 0x00ff00,
        footer: {
          text: 'For real ML training, use a dedicated ML platform'
        }
      }],
      components: [{
        type: 1, // Action Row
        components: [{
          type: 2, // Button
          style: 1, // Primary
          label: '🔗 Learn More',
          custom_id: 'learn_more_ml'
        }, {
          type: 2,
          style: 2, // Secondary
          label: '📊 View Alternatives',
          custom_id: 'view_alternatives'
        }]
      }],
      flags: 64 // Ephemeral
    }
  });
}

async function handleLinearRegressionCommand(interaction, env) {
  // Simulate linear regression without heavy libraries
  const sampleData = generateSampleData();
  
  return jsonResponse({
    type: InteractionResponseType.CHANNEL_MESSAGE_WITH_SOURCE,
    data: {
      embeds: [{
        title: '📈 Linear Regression Simulation',
        description: 'Generated sample data and calculated regression line:',
        fields: [
          {
            name: '📊 Sample Data',
            value: `Points: ${sampleData.points}\nSlope: ${sampleData.slope}\nIntercept: ${sampleData.intercept}`,
            inline: true
          },
          {
            name: '📐 Equation',
            value: `y = ${sampleData.slope}x + ${sampleData.intercept}`,
            inline: true
          },
          {
            name: '🎯 R² Score',
            value: `${sampleData.r_squared}`,
            inline: true
          }
        ],
        color: 0x0099ff,
        footer: {
          text: 'This is a mathematical simulation - no actual plotting available in Workers'
        }
      }],
      components: [{
        type: 1,
        components: [{
          type: 2,
          style: 1,
          label: '🔄 Generate New Data',
          custom_id: 'regenerate_data'
        }, {
          type: 2,
          style: 3, // Success
          label: '📊 Export Data',
          custom_id: 'export_data'
        }]
      }],
      flags: 64
    }
  });
}

async function handleComponent(interaction, env) {
  const customId = interaction.data.custom_id;

  switch (customId) {
    case 'learn_more_ml':
      return jsonResponse({
        type: InteractionResponseType.CHANNEL_MESSAGE_WITH_SOURCE,
        data: {
          content: '🔗 **ML Resources:**\n\n• **TensorFlow.js**: Run ML in the browser\n• **Google Colab**: Free GPU training\n• **Hugging Face**: Pre-trained models\n• **AWS SageMaker**: Cloud ML platform\n• **Replicate**: ML API service',
          flags: 64
        }
      });

    case 'view_alternatives':
      return jsonResponse({
        type: InteractionResponseType.CHANNEL_MESSAGE_WITH_SOURCE,
        data: {
          content: '📊 **ML Alternatives for Discord Bots:**\n\n• **External APIs**: Use services like OpenAI, Hugging Face\n• **Serverless Functions**: Deploy on Vercel/Netlify with ML libraries\n• **Dedicated Servers**: VPS with full Python ML stack\n• **Hybrid Approach**: Workers for bot logic + external ML service',
          flags: 64
        }
      });

    case 'regenerate_data':
      const newData = generateSampleData();
      return jsonResponse({
        type: InteractionResponseType.UPDATE_MESSAGE,
        data: {
          embeds: [{
            title: '📈 Linear Regression Simulation (Updated)',
            description: 'Generated new sample data and calculated regression line:',
            fields: [
              {
                name: '📊 Sample Data',
                value: `Points: ${newData.points}\nSlope: ${newData.slope}\nIntercept: ${newData.intercept}`,
                inline: true
              },
              {
                name: '📐 Equation',
                value: `y = ${newData.slope}x + ${newData.intercept}`,
                inline: true
              },
              {
                name: '🎯 R² Score',
                value: `${newData.r_squared}`,
                inline: true
              }
            ],
            color: 0x0099ff,
            footer: {
              text: 'This is a mathematical simulation - no actual plotting available in Workers'
            }
          }],
          components: [{
            type: 1,
            components: [{
              type: 2,
              style: 1,
              label: '🔄 Generate New Data',
              custom_id: 'regenerate_data'
            }, {
              type: 2,
              style: 3,
              label: '📊 Export Data',
              custom_id: 'export_data'
            }]
          }]
        }
      });

    case 'export_data':
      return jsonResponse({
        type: InteractionResponseType.CHANNEL_MESSAGE_WITH_SOURCE,
        data: {
          content: '📊 **Data Export Feature:**\n\nIn a full implementation, this would:\n• Generate CSV data\n• Create downloadable links\n• Send data via DM\n\n*Currently simulated due to Workers limitations*',
          flags: 64
        }
      });

    default:
      return jsonResponse({
        type: InteractionResponseType.CHANNEL_MESSAGE_WITH_SOURCE,
        data: {
          content: 'Unknown button interaction!',
          flags: 64
        }
      });
  }
}

function generateSampleData() {
  // Simple linear regression simulation
  const points = Math.floor(Math.random() * 50) + 10;
  const slope = (Math.random() * 4 - 2).toFixed(2);
  const intercept = (Math.random() * 20 - 10).toFixed(2);
  const r_squared = (0.7 + Math.random() * 0.25).toFixed(3);

  return {
    points,
    slope,
    intercept,
    r_squared
  };
}

function jsonResponse(data) {
  return new Response(JSON.stringify(data), {
    headers: { 'Content-Type': 'application/json' },
  });
}

// Command registration function (run this once to register commands)
export async function registerCommands(env) {
  const commands = [
    {
      name: 'ping',
      description: 'Test if the bot is running'
    },
    {
      name: 'create_neural_network',
      description: 'Simulate neural network creation (educational purpose)'
    },
    {
      name: 'graph_linear_regression',
      description: 'Simulate linear regression calculation'
    }
  ];

  const url = `${DISCORD_API_URL}/applications/${env.DISCORD_APPLICATION_ID || DISCORD_APPLICATION_ID}/commands`;
  
  const response = await fetch(url, {
    method: 'PUT',
    headers: {
      'Authorization': `Bot ${env.DISCORD_TOKEN || DISCORD_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(commands),
  });

  if (response.ok) {
    console.log('Commands registered successfully');
  } else {
    console.error('Failed to register commands:', await response.text());
  }
}